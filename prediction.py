import time, math, random
import numpy as np
import torch
import inspect
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int, deterministic: bool = True):
    import numpy as _np, random as _random, torch as _torch
    _random.seed(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)
    _torch.cuda.manual_seed_all(seed)
    if deterministic:
        _torch.backends.cudnn.deterministic = True
        _torch.backends.cudnn.benchmark = False


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TimeSeriesDataset(Dataset):

    def __init__(self, data, timesteps, horizon, stride):
        assert stride >= 1, "stride 必须为正整数"
        self.timesteps = timesteps
        self.horizon = horizon
        self.stride = stride
        self.data = data

        x_list, y_list = [], []
        end = len(data) - timesteps - horizon + 1
        for i in range(0, end, stride):
            x_seq = data[i : i + timesteps, -1]
            y_seq = data[i + timesteps : i + timesteps + horizon, -1]
            x_list.append(x_seq)
            y_list.append(y_seq)
        self.features = torch.tensor(np.array(x_list), dtype=torch.float32)
        self.targets  = torch.tensor(np.array(y_list), dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def data_standarlization(train_data, test_data, fi=-1):
    scaler = StandardScaler()
    scaler.fit(train_data)
    std_train = scaler.transform(train_data)
    std_test  = scaler.transform(test_data)
    return std_train, std_test, scaler.mean_[fi], scaler.scale_[fi]


def inverse_standarlization(x, mean, std):
    return x * std + mean


class AttentionRecursive(nn.Module):
    def __init__(self, input_dim, hidden_dim, horizon, num_layers, dropout, attn_dim):
        super().__init__()
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.attn_dim = hidden_dim if attn_dim is None else int(attn_dim)

        self.input_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attn = nn.Linear(hidden_dim * 2, self.attn_dim)
        self.v = nn.Parameter(torch.randn(self.attn_dim))

        self.recursive = nn.LSTMCell(1, hidden_dim)
        self.forecast = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, y=None, teacher_forcing_ratio=1.0):
        B, T, F = x.size()
        x = self.input_dropout(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        h_dec, c_dec = h_n[-1], c_n[-1]

        recur_in = x.new_zeros(B, 1)
        outputs = []
        for t in range(self.horizon):
            h_rep = h_dec.unsqueeze(1).repeat(1, T, 1)
            attn_in = torch.cat([lstm_out, h_rep], dim=2)
            energy = torch.tanh(self.attn(attn_in))
            scores = energy.matmul(self.v)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            context = (lstm_out * weights).sum(dim=1)

            h_dec, c_dec = self.recursive(recur_in, (h_dec, c_dec))
            fuse = torch.cat([h_dec, context], dim=-1)
            y_t = self.forecast(fuse)
            outputs.append(y_t)

            scheduled_sampling = torch.rand(1).item() < teacher_forcing_ratio
            if scheduled_sampling:
                recur_in = y[:, t:t + 1]
            else:
                recur_in = y_t

        return torch.cat(outputs, dim=1)  # [B, H]


def exponent_scheduled_sampling(epoch, alpha):

    p_tf = alpha ** epoch
    return float(max(min(p_tf, 1.0), 0.0))


def train_model(train_loader, model, criterion, optimizer, scheduler,
                device, epochs: int, alpha_ss: float, increase_rate: float, decrease_rate: float):
    model.train()
    epoch_latencies = []
    previous_loss = float('inf')
    tfr = 1.01
    loss = 0.0

    for ep in range(epochs):
        start_epoch_time = time.perf_counter()
        teacher_ratio = exponent_scheduled_sampling(ep, alpha_ss)
        total = 0.0
        steps = 0

        for x, y in train_loader:
            x = x.unsqueeze(-1).to(device)
            y = y.to(device)

            step_start_time = time.perf_counter()
            optimizer.zero_grad()

            if previous_loss < loss:
                tfr = min(tfr + increase_rate, 1)
            else:
                tfr = min(teacher_ratio, tfr - decrease_rate)

            out = model(x, y, tfr)
            loss = criterion(out, y)
            previous_loss = loss.item()

            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)

            step_end_time = time.perf_counter()
            step_latency = (step_end_time - step_start_time) * 1000
            steps += 1

        scheduler.step()

        epoch_end_time = time.perf_counter()
        epoch_latency = (epoch_end_time - start_epoch_time) * 1000
        epoch_latencies.append(epoch_latency)

        avg_step_latency = epoch_latency / steps

        print(f"[AttentionRecur] Epoch {ep+1}/{epochs} | TFR={tfr:.2f} | TrainLoss={(total/len(train_loader.dataset)):.4f}"
              f"| Avg Step Latency: {avg_step_latency:.4f} ms/step | Epoch Latency: {epoch_latency:.4f} ms")

    avg_epoch_latency = sum(epoch_latencies) / len(epoch_latencies)
    print(f"Average Epoch Latency: {avg_epoch_latency:.4f} ms")
    return avg_epoch_latency


def predict_rolling_window(model, test_loader, std_mean, std_std, device,
                           return_latency: bool = True):

    model.eval()
    preds, trues = [], []

    lat_samples, lat_total_ms, lat_steps = 0, 0.0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.unsqueeze(-1).to(device)
            y = y.to(device)

            if return_latency:
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
            out = model(x, None, 0.0)
            if return_latency:
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1000.0
                lat_total_ms += dt
                lat_steps += (x.size(0) * out.size(1))
                lat_samples += x.size(0)

            preds.append(out.cpu())
            trues.append(y.cpu())

    predictions = torch.cat(preds, dim=0).numpy()
    actuals     = torch.cat(trues, dim=0).numpy()

    pred_out = inverse_standarlization(predictions.reshape(-1), std_mean, std_std)
    true_out = inverse_standarlization(actuals.reshape(-1),     std_mean, std_std)

    if return_latency and lat_steps > 0:
        ms_per_step = lat_total_ms / lat_steps
    else:
        ms_per_step = None

    return pred_out, true_out, ms_per_step


def estimate_flops_attention_recursive(T: int, H: int, m: int, h: int, L: int, attn_dim: int) -> int:
    a = h if attn_dim is None else int(attn_dim)

    flops_enc_first = 2 * 4 * (m * h + h * h) * T
    flops_enc_rest  = 0
    for _ in range(L - 1):
        flops_enc_rest += 2 * 4 * (h * h + h * h) * T
    flops_enc = flops_enc_first + flops_enc_rest

    flops_attn_per_dec = (2 * h * a + a) * T + 2 * a * T
    flops_attn_per_dec += 5 * T

    flops_dec_per_step = 2 * 4 * (1 * h + h * h) + 2 * (2 * h)

    flops_total = flops_enc + H * (flops_attn_per_dec + flops_dec_per_step)
    return int(flops_total)


def estimate_vram_inference_bytes(B: int, T: int, H: int, h: int, dtype_bytes: int = 4) -> int:
    activ = B * T * h + B * h + B * H
    return int(activ * dtype_bytes)


def estimate_vram_bytes_full(
    model,
    *,
    B: int, T: int, H: int,
    m: int, h: int, L: int,
    attn_dim: int,
    mode: str = "infer",        # "infer" 或 "train"
    dtype: str = "fp32",        # "fp32" | "fp16" | "bf16"
    optimizer: str = "adam",   # "adamw" | "adam" | "sgd_m" | "sgd"
    amp: bool = False,
    grad_ckpt: bool = False,

    cuda_context_mb: int = 400,
    cudnn_ws_mb: int = 256,
    allocator_headroom: float = 0.12,
    pin_mem_mb: float = 0.0,
    buffers_mb: int = 64
):

    bytes_map = {"fp16": 2, "bf16": 2, "fp32": 4}
    param_dtype_bytes = bytes_map.get(dtype, 4)
    grad_bytes = 4
    master_bytes = 4

    P = count_trainable_parameters(model)
    param_b = P * param_dtype_bytes

    grad_b = P * grad_bytes if mode == "train" else 0
    if mode == "train":
        if optimizer in ("adam", "adamw"):
            opt_b = P * (4 + 4)
            master_w_b = P * master_bytes if amp else 0
        elif optimizer == "sgd_m":
            opt_b = P * 4
            master_w_b = P * master_bytes if amp else 0
        else:
            opt_b = 0
            master_w_b = P * master_bytes if amp else 0
    else:
        opt_b = 0
        master_w_b = 0

    dtype_bytes = param_dtype_bytes if not amp else 2
    enc_out_b = B * T * h * dtype_bytes
    attn_mid_b = B * T * attn_dim * dtype_bytes * H
    attn_wgt_b = B * T * dtype_bytes * H
    dec_b      = B * H * h * dtype_bytes
    out_b      = B * H * dtype_bytes
    activ_infer_b = enc_out_b + attn_mid_b + attn_wgt_b + dec_b + out_b

    if mode == "train":
        k = 8
        activ_train_extra = (B * T * L * h * k + B * H * h * k) * dtype_bytes
    else:
        activ_train_extra = 0

    activ_b = activ_infer_b + activ_train_extra
    if grad_ckpt and mode == "train":
        activ_b = int(activ_b * 0.6)

    overhead_b = int((cuda_context_mb + cudnn_ws_mb + buffers_mb) * 1024 * 1024)
    pin_b      = int(pin_mem_mb * 1024 * 1024)

    core_b = param_b + grad_b + master_w_b + opt_b + activ_b
    core_b = int(core_b * (1.0 + allocator_headroom))

    total_b = core_b + overhead_b + pin_b
    breakdown = {
        "params": param_b, "grads": grad_b, "master_w": master_w_b, "opt_states": opt_b,
        "activations": activ_b, "allocator_headroom": int(core_b - (param_b + grad_b + master_w_b + opt_b + activ_b)),
        "cuda_context": int(cuda_context_mb * 1024 * 1024),
        "cudnn_workspace": int(cudnn_ws_mb * 1024 * 1024),
        "buffers": int(buffers_mb * 1024 * 1024),
        "pin_memory": pin_b,
        "total": total_b
    }
    return total_b, breakdown


def measure_vram_peak_generic(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    train: bool,
    use_amp: bool = False,
    optimizer_name: str = "adam",
    lr: float = 1e-3,
    classify_loss_fn: callable
) -> dict:

    device = x.device
    if device.type != "cuda":
        return {"peak_reserved": 0, "peak_allocated": 0, "delta_allocated": 0}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base_alloc = torch.cuda.memory_allocated(device)

    if train:
        model.train()
        if optimizer_name == "adamw":
            opt = torch.optim.AdamW(model.parameters(), lr=lr)
        elif optimizer_name == "adam":
            opt = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "sgd_m":
            opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            opt = torch.optim.SGD(model.parameters(), lr=lr)

        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        if y is None:
            crit = nn.MSELoss()
            with torch.cuda.amp.autocast(enabled=use_amp):
                sig = inspect.signature(model.forward)
                kw = {}
                if 'y' in sig.parameters:
                    kw['y'] = y if y is not None else torch.zeros((x.size(0), getattr(model, 'horizon', 1)),
                                                                  device=x.device, dtype=x.dtype)
                if 'teacher_forcing_ratio' in sig.parameters:
                    kw['teacher_forcing_ratio'] = 0.5
                out = model(x, **kw)

                y_ = torch.zeros_like(out, device=out.device)
                loss = crit(out, y_)
        else:
            if classify_loss_fn is not None:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    loss = classify_loss_fn(model, x, y)
            else:
                crit = nn.MSELoss()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(x, y, 0.5) if hasattr(model, "horizon") else model(x)
                    loss = crit(out, y)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    else:
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            sig = inspect.signature(model.forward)
            kw = {}
            if 'y' in sig.parameters:
                kw['y'] = None
            if 'teacher_forcing_ratio' in sig.parameters:
                kw['teacher_forcing_ratio'] = 0.0
            _ = model(x, **kw)

    torch.cuda.synchronize(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    peak_alloc    = torch.cuda.max_memory_allocated(device)
    return {
        "peak_reserved": int(peak_reserved),
        "peak_allocated": int(peak_alloc),
        "delta_allocated": int(peak_alloc - base_alloc)
    }


def measure_cuda_vram_runtime(model: nn.Module, x_ex: torch.Tensor) -> tuple[int, int]:
    if x_ex.device.type != 'cuda':
        return 0, 0
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.no_grad():
        _ = model(x_ex, None, 0.0)
    peak = torch.cuda.max_memory_allocated()
    return int(peak), int(peak - base)