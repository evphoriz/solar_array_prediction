import torch
import math
import time
import inspect
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def set_seed(seed: int, deterministic: bool = True):
    import numpy as _np, random as _random, torch as _torch
    _random.seed(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)
    _torch.cuda.manual_seed_all(seed)
    if deterministic:
        _torch.backends.cudnn.deterministic = True
        _torch.backends.cudnn.benchmark = False


def data_build(X, y, timesteps, horizon, stride):
    assert stride >= 1, "stride 必须为正整数"
    x_data = []
    y_data = []
    end = len(X) - timesteps - horizon + 1
    for i in range(0, end, stride):
        seq_x = X[i : i + timesteps, :]
        seq_y = y[i + timesteps : i + timesteps + horizon]
        x_data.append(seq_x)
        y_data.append(seq_y)
    X_out = np.stack(x_data, axis=0)
    y_out = np.stack(y_data, axis=0)
    return X_out, y_out


def split_train_test(X, label, target, ratio_train=0.8):
    split_idx = int(len(X) * ratio_train)
    X_train, label_train, target_train = X[:split_idx], label[:split_idx], target[:split_idx]
    X_test, label_test, target_test = X[split_idx:], label[split_idx:], target[split_idx:]

    return X_train, label_train, target_train, X_test, label_test, target_test


def data_standarlization(train_data, test_data, fi=-1):
    scalar = StandardScaler()
    scalar.fit(train_data)
    std_train_data = scalar.transform(train_data)
    std_test_data = scalar.transform(test_data)
    std_mean = scalar.mean_[fi]
    std_std = scalar.scale_[fi]
    return std_train_data, std_test_data, std_mean, std_std


def inverse_standarlization(preds, mean, std):
    return preds * std + mean


def load_downsample(rout_files, mane_files, num_sk=3036):
    combined_data_rout = []
    combined_data_mane = []
    data_column_south = ['南太阳电池阵温度1', '南太阳电池阵温度2', '南太阳电池阵温度3',
                         '光照区/阴影区', '南太阳阵1电压', '南太阳阵2电压',
                         '南太阳阵5电压', '南太阳阵6电压',
                         '南太阳阵7电压', '工况', '南太阳阵电流']

    data_column_north = ['北太阳电池温度1', '北太阳电池阵温度2', '北太阳电池阵温度3',
                         '光照区/阴影区', '北太阳阵1电压', '北太阳阵2电压',
                         '北太阳阵5电压', '北太阳阵6电压',
                         '北太阳阵7电压',  '工况', '北太阳阵电流']


    for file in rout_files:
        dataframes = pd.read_csv(file)
        data_south = dataframes[data_column_south]
        X_south = data_south.drop(columns=['工况', '南太阳阵电流']).values
        label_south = data_south['工况'].values
        target_south = data_south['南太阳阵电流'].values

        data_north = dataframes[data_column_north]
        X_north = data_north.drop(columns=['工况', '北太阳阵电流']).values
        label_north = data_north['工况'].values
        target_north = data_north['北太阳阵电流'].values

        X = np.concatenate((X_south, X_north), axis=0)
        label = np.concatenate((label_south, label_north), axis=0)
        target = np.concatenate((target_south, target_north), axis=0)

        label_is_indices = np.where(label == 2)[0]
        if len(label_is_indices) == 0:
            continue

        p = len(label_is_indices)

        label_lg_indices = np.where(label == 0)[0]
        label_es_indices = np.where(label == 1)[0]
        label_os_indices = np.where(label == 3)[0]


        first_label_is_index = label_is_indices[0]
        if len(label_lg_indices) > p:
            interval_lg = len(label_lg_indices) // p
            sampled_label_lg_before = label_lg_indices[:len(label_lg_indices) - (len(label_lg_indices) % interval_lg)][::interval_lg][:p]
        else:
            sampled_label_lg_before = label_lg_indices[:first_label_is_index]

        last_label_os_index = label_os_indices[-1]
        if len(label_lg_indices) > p:
            out_lg_indices = last_label_os_index - len(label_es_indices) - len(label_is_indices) - len(label_os_indices)
            interval_es = (len(label_lg_indices) - (out_lg_indices + 1)) // p
            if interval_es > 0:
                sampled_label_lg_after = label_lg_indices[out_lg_indices + 1:][::interval_es][:p]
            else:
                sampled_label_lg_after = label_lg_indices[last_label_os_index + 1:][:p]
        else:
            sampled_label_lg_after = label_lg_indices[last_label_os_index + 1:]

        sampled_label_es_indices = label_es_indices[np.linspace(0, len(label_es_indices) - 1, 2 * p, dtype=int)]


        final_indices = np.concatenate(
            (sampled_label_lg_before, label_is_indices, sampled_label_es_indices, label_os_indices, sampled_label_lg_after))


        final_features = X[final_indices]
        final_labels = label[final_indices]
        final_targets = target[final_indices]

        combined_data_rout.append((final_features, final_labels, final_targets))

    eq_features = np.vstack([data[0] for data in combined_data_rout])
    eq_labels = np.hstack([data[1] for data in combined_data_rout])
    eq_targets = np.hstack([data[2] for data in combined_data_rout])

    X_eq_train, label_eq_train, target_eq_train, X_eq_test, label_eq_test, target_eq_test = split_train_test(eq_features, eq_labels, eq_targets)


    for file in mane_files:
        dataframes = pd.read_csv(file)
        data_south = dataframes[data_column_south]
        X_south = data_south.drop(columns=['工况', '南太阳阵电流']).values
        label_south = data_south['工况'].values
        target_south = data_south['南太阳阵电流'].values

        data_north = dataframes[data_column_north]
        X_north = data_north.drop(columns=['工况', '北太阳阵电流']).values
        label_north = data_north['工况'].values
        target_north = data_north['北太阳阵电流'].values

        X = np.concatenate((X_south, X_north), axis=0)
        label = np.concatenate((label_south, label_north), axis=0)
        target = np.concatenate((target_south, target_north), axis=0)


        label_sk_indices = np.where(label == 4)[0]
        if len(label_sk_indices) == 0:
            continue

        sampled_label_sk_indices = label_sk_indices[np.linspace(0, len(label_sk_indices) - 1, num_sk, dtype=int)]

        final_features = X[sampled_label_sk_indices]
        final_labels = label[sampled_label_sk_indices]
        final_targets = target[sampled_label_sk_indices]

        combined_data_mane.append((final_features, final_labels, final_targets))


    sk_features = np.vstack([data[0] for data in combined_data_mane])
    sk_labels = np.hstack([data[1] for data in combined_data_mane])
    sk_targets = np.hstack([data[2] for data in combined_data_mane])

    X_sk_train, label_sk_train, target_sk_train, X_sk_test, label_sk_test, target_sk_test = split_train_test(sk_features, sk_labels, sk_targets)


    X_train = np.concatenate((X_eq_train, X_sk_train), axis=0)
    label_train = np.concatenate((label_eq_train, label_sk_train), axis=0)
    target_train = np.concatenate((target_eq_train, target_sk_train), axis=0)

    X_test = np.concatenate((X_eq_test, X_sk_test), axis=0)
    label_test = np.concatenate((label_eq_test, label_sk_test), axis=0)
    target_test = np.concatenate((target_eq_test, target_sk_test), axis=0)

    return X_train, label_train, target_train, X_test, label_test, target_test


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, horizon, dropout=0.1):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizon = horizon

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x):

        B = x.size(0)
        h0 = torch.zeros(self.num_layers, B, self.hidden_dim, device=x.device)

        output, hn = self.gru(x, h0)
        last = output[:, -1, :]
        y = self.fc(last)
        return y


def data_loader(batch_size, X_train, y_train, X_test, y_test, X_val=None, y_val=None):
    train_dataset = torch.tensor(X_train, dtype=torch.float32)
    train_target = torch.tensor(y_train, dtype=torch.float32)

    test_dataset = torch.tensor(X_test, dtype=torch.float32)
    test_target = torch.tensor(y_test, dtype=torch.float32)

    train_id = TensorDataset(train_dataset, train_target)
    test_id = TensorDataset(test_dataset, test_target)

    train_loader = DataLoader(dataset=train_id, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_id, batch_size=batch_size, shuffle=False)

    if X_val is not None or y_val is not None:
        val_dataset = torch.tensor(X_val, dtype=torch.float32)
        val_target = torch.tensor(y_val, dtype=torch.float32)

        val_id = TensorDataset(val_dataset, val_target)
        val_loader = DataLoader(dataset=val_id, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader, val_loader
    else:
        return train_loader, test_loader


class TimeSeriesDataset(Dataset):
    def __init__(self, data, lag=10):
        self.lag = lag

        features_list = []
        targets_list = []
        for i in range(len(data) - lag):
            feature_row = data[i:i + lag ,-1]
            features_list.append(feature_row)
            targets_list.append(data[i + lag ,-1])

        self.features = torch.tensor(features_list, dtype=torch.float32)
        self.targets = torch.tensor(targets_list, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops_gru(
    seq_len: int,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    output_dim: int,
    bidirectional: bool = False,
    include_elementwise: bool = False,
) -> int:

    dirs = 2 if bidirectional else 1
    total_flops = 0
    in_dim = input_dim

    for layer in range(num_layers):

        flops_per_step_per_dir = 2 * 3 * (in_dim * hidden_dim + hidden_dim * hidden_dim)
        if include_elementwise:
            flops_per_step_per_dir += 10 * hidden_dim


        layer_flops = seq_len * dirs * flops_per_step_per_dir
        total_flops += layer_flops

        in_dim = hidden_dim * dirs

    d_in = hidden_dim * dirs
    fc_flops = 2 * d_in * output_dim
    total_flops += fc_flops

    return int(total_flops)


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
                # 回归
                crit = nn.MSELoss()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(x)
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


def train_model(train_loader, model, criterion, optimizer, device, epochs):
    """ 训练模型函数 """
    model.train()
    for epoch in range(epochs):

        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss={(total_loss / len(train_loader.dataset)):.4f}")


def test_model(model, test_loader, std_mean, std_std, device, return_latency: bool = True):

    model.eval()
    predictions = []
    target_list = []

    lat_samples = 0
    lat_total_ms = 0.0
    lat_steps =  0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            if return_latency:
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

            preds = model(inputs)
            if return_latency:
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1000.0
                lat_total_ms += dt
                lat_steps += (inputs.size(0) * preds.size(1))
                lat_samples += inputs.size(0)

            output_inverse = inverse_standarlization(preds, std_mean, std_std)
            target_inverse = inverse_standarlization(targets, std_mean, std_std)
            predictions.extend(output_inverse.cpu().numpy())
            target_list.extend(target_inverse.cpu().numpy())

    pred_seq = np.array(predictions)[:,-1]
    target_seq = np.array(target_list)[:,-1]

    if return_latency and lat_steps > 0:
        ms_per_step = lat_total_ms / lat_steps
    else:
        ms_per_step = None

    return pred_seq, target_seq, ms_per_step


if __name__ == "__main__":
    rout_files = ['data/Y2101_p.csv','data/Y2201_p.csv','data/Y2202_p.csv',
                  'data/Y2301_p.csv','data/Y2302_p.csv','data/Y2401_p.csv']
    mane_files = ['data/Y2203_p.csv','data/Y2402_p.csv']

    X_train, _, target_train, X_test, _, target_test = load_downsample(rout_files, mane_files)

    train_sum = np.concatenate((X_train, target_train.reshape(-1, 1)), axis=1)
    test_sum = np.concatenate((X_test, target_test.reshape(-1, 1)), axis=1)

    # 模型参数
    USE_AMP = False
    OPT_KIND = "adam"
    batch_size = 128
    num_epochs = 100
    time_steps = 20
    horizon = 10
    stride = 10
    dropout = 0.1
    learning_rate = 0.001
    bidirectional = False

    hidden_dim = 512
    layers_num = 7
    input_dim = X_train.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_save_dir = 'trained_rolling_model'

    # 实例化模型、损失函数以及优化器
    model = GRUModel(input_dim, hidden_dim, layers_num, horizon, dropout).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    seeds = [2025, 2026, 2027, 2028, 2029]

    mae_list = []
    mse_list = []
    rmse_list = []
    r2_list = []
    lat_reg_list = []

    for run_id, seed in enumerate(seeds, 1):
        print(f"\n========== Run {run_id}/5 | Seed={seed} ==========")
        set_seed(seed)

        std_train_data, std_test_data, std_mean, std_std = data_standarlization(train_sum, test_sum)

        X_train_seq, y_train_seq = data_build(std_train_data[:, :-1], std_train_data[:, -1], time_steps, horizon, stride)
        train_id = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32),
                                 torch.tensor(y_train_seq, dtype=torch.float32))
        dataloader_train = DataLoader(train_id, batch_size=batch_size, shuffle=True)

        X_test_seq, y_test_seq = data_build(std_test_data[:, :-1], std_test_data[:, -1], time_steps, horizon, stride)
        test_id = TensorDataset(torch.tensor(X_test_seq, dtype=torch.float32),
                                torch.tensor(y_test_seq, dtype=torch.float32))
        dataloader_test = DataLoader(test_id, batch_size=batch_size, shuffle=False)

        if device.type == 'cuda':
            xb_te, yb_te = next(iter(dataloader_test))
            xb_tr, yb_tr = next(iter(dataloader_train))
            xb_te = xb_te.to(device)
            yb_te = yb_te.to(device)
            xb_tr = xb_tr.to(device)
            yb_tr = yb_tr.to(device)

            probe_reg_train = measure_vram_peak_generic(
                model, xb_tr, yb_tr, train=True, use_amp=USE_AMP,
                optimizer_name=OPT_KIND, lr=learning_rate, classify_loss_fn=None)

            probe_reg_infer = measure_vram_peak_generic(
                model, xb_te, None, train=False, use_amp=USE_AMP, classify_loss_fn=None)

            vram_train_reserved = probe_reg_train["peak_reserved"]
            vram_infer_reserved = probe_reg_infer["peak_reserved"]
        else:
            continue


        train_model(dataloader_train, model, criterion, optimizer, device, num_epochs)
        pred_seq, actual_seq, ms_per_step = test_model(model, dataloader_test, std_mean, std_std, device)

        flops_gru = estimate_flops_gru(time_steps, input_dim, hidden_dim, layers_num, horizon, bidirectional)

        print("可训练总参数量:", count_trainable_parameters(model))
        print(f"总FLOPs：{flops_gru:,} FLOPs")
        print(f"[VRAM] 推理合计 ≈ {vram_infer_reserved / 1024 / 1024:.1f} MB")
        print(f"[VRAM] 训练合计 ≈ {vram_train_reserved / 1024 / 1024:.1f} MB")


        test_mae = mean_absolute_error(actual_seq, pred_seq)
        test_mse = mean_squared_error(actual_seq, pred_seq)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(actual_seq, pred_seq)

        mae_list.append(test_mae)
        mse_list.append(test_mse)
        rmse_list.append(test_rmse)
        r2_list.append(test_r2)

        if ms_per_step is not None:
            lat_reg_list.append(ms_per_step)

    def avg(x):
        return float(np.mean(x)) if len(x)>0 else float('nan')

    print(f"MAE: {avg(mae_list):.6f}, RMSE: {avg(rmse_list):.6f}, R²: {avg(r2_list):.4f}")
    print(f"Latency  (ms/step): {avg(lat_reg_list):.4f}")
    print(f"Throughput  (num/s): {1000/ (avg(lat_reg_list)):.4f}")