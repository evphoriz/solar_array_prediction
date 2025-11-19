import os, time, random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd


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


def load_downsample(rout_files, mane_files, num_sk=3036):
    combined_data_rout = []
    combined_data_mane = []

    # --------------------------10个维度特征-------------------------------
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


class Chomp(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride, dil, pad, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, stride=stride, padding=pad, dilation=dil)
        self.chomp1 = Chomp(pad)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.dp1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, stride=stride, padding=pad, dilation=dil)
        self.chomp2 = Chomp(pad)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dp2 = nn.Dropout(dropout)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.dp1(self.relu1(self.chomp1(self.conv1(x))))
        y = self.dp2(self.relu2(self.chomp2(self.conv2(y))))
        res = x if self.down is None else self.down(x)
        return self.relu(y + res)

class DTCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout, num_classes, horizon):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dil = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            pad = (kernel_size - 1) * dil
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, 1, dil, pad, dropout))
        self.net = nn.Sequential(*layers)
        self.horizon = horizon
        self.num_classes = num_classes
        self.fc = nn.Linear(num_channels[-1], num_classes * horizon)
    def forward(self, x):
        x = x.transpose(1,2)
        y = self.net(x)
        y = y[:, :, -1]
        y = self.fc(y)
        return y.view(y.size(0), self.horizon, self.num_classes)


def aggregate_over_time(logits, mode, alpha=None):

    B, H, C = logits.shape
    if mode == 'ordered1':
        per_class, _ = logits.max(dim=1)

        return per_class, None

    elif mode == 'ordered2':
        probs = torch.softmax(logits, dim=2)

        if alpha is None:
            w = torch.full((H,), 1.0 / H, device=logits.device, dtype=logits.dtype)

        else:
            if (torch.is_tensor(alpha) and alpha.ndim == 1) or isinstance(alpha, (list, tuple)):
                w = torch.as_tensor(alpha, device=logits.device, dtype=logits.dtype).view(-1)

            elif torch.is_tensor(alpha) and alpha.ndim == 0 or isinstance(alpha, (float, int)):
                a = float(alpha.item()) if torch.is_tensor(alpha) else float(alpha)
                exps = torch.arange(H - 1, -1, -1, device=logits.device, dtype=logits.dtype)
                base = torch.tensor(a, device=logits.device, dtype=logits.dtype)
                w = torch.pow(base, exps)

            else:
                raise ValueError("alpha must be None, a scalar, or a length-H 1D vector.")

            if w.numel() != H:
                raise ValueError(f"alpha length mismatch: expected H={H}, got {w.numel()}")

            w = w / (w.sum() + 1e-12)

        per_class = (probs * w.view(1, H, 1)).sum(dim=1)
        per_class = per_class / (per_class.sum(dim=1, keepdim=True) + 1e-12)

        classes = torch.arange(C, device=logits.device, dtype=per_class.dtype)
        y_pred_real = (per_class * classes.view(1, C)).sum(dim=1)

        return per_class, y_pred_real

    else:
        raise ValueError("mode must be 'ordered1' or 'ordered2'")


def train_model(train_loader, model, epochs, criterion, optimizer, scheduler,
                device, decision_mode='ordered1', alpha=None):
    model.train()
    epoch_latencies = []

    for ep in range(epochs):
        start_epoch_time = time.perf_counter()
        tot, corr, tot_n = 0.0, 0, 0
        steps = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            step_start_time = time.perf_counter()
            optimizer.zero_grad()

            out = model(x)
            per_class,_ = aggregate_over_time(out, decision_mode, alpha)
            loss = criterion(per_class, y.long())

            loss.backward()
            optimizer.step()
            _, pred = per_class.max(dim=1)
            tot += loss.item()
            corr += (pred==y).sum().item()
            tot_n += y.size(0)

            step_end_time = time.perf_counter()
            step_latency = (step_end_time - step_start_time) * 1000
            steps += 1

        scheduler.step()

        epoch_end_time = time.perf_counter()
        epoch_latency = (epoch_end_time - start_epoch_time) * 1000
        epoch_latencies.append(epoch_latency)

        avg_step_latency = epoch_latency / steps

        print(f"[Classify] Epoch {ep+1}/{epochs} | Loss {tot/len(train_loader):.4f} | Acc {corr/tot_n:.4f}"
              f"| Avg Step Latency: {avg_step_latency:.4f} ms/step | Epoch Latency: {epoch_latency:.4f} ms")

    avg_epoch_latency = sum(epoch_latencies) / len(epoch_latencies)
    print(f"Average Epoch Latency: {avg_epoch_latency:.4f} ms")
    return avg_epoch_latency


def predict_rolling_classify(model, test_loader, criterion, device, decision_mode='ordered1', alpha=None,
                              return_latency: bool = True):

    model.eval()
    preds, labels = [], []
    lat_total_ms, lat_steps = 0.0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            if return_latency and device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(x)
            if return_latency and device.type == 'cuda':
                torch.cuda.synchronize()
            lat_total_ms += (time.perf_counter() - t0) * 1000.0
            lat_steps += x.size(0) * out.size(1)  # 样本×H

            per_class,_ = aggregate_over_time(out, decision_mode, alpha)
            pred = per_class.argmax(dim=1)
            preds.append(pred.cpu())
            labels.append(y.cpu())

    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(labels).numpy()
    ms_per_step = lat_total_ms / lat_steps if return_latency and lat_steps>0 else None
    return y_pred, y_true, ms_per_step


def estimate_flops_tcn(num_inputs, num_channels, kernel_size, T):
    L = T
    flops = 0
    cin = num_inputs
    for cout in num_channels:
        flops += 2 * cout * L * (cin * kernel_size)
        cin = cout
    return int(flops)