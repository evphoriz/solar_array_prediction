import os, random, numpy as np, torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from perception import (
    data_build, load_downsample, split_train_test,
    DTCN, train_model as train_model_cls, predict_rolling_classify,
    estimate_flops_tcn, aggregate_over_time
)
from prediction import (
    TimeSeriesDataset, data_standarlization,
    AttentionRecursive, train_model as train_model_reg, predict_rolling_window,
    set_seed, count_trainable_parameters,
    estimate_flops_attention_recursive, estimate_vram_inference_bytes, estimate_vram_bytes_full, measure_vram_peak_generic
)


def uniform_downsample(uniform_files, n):
    all_data = []

    # --------------------------10个维度特征-------------------------------
    data_column_south = ['南太阳电池阵温度1', '南太阳电池阵温度2', '南太阳电池阵温度3',
                         '光照区/阴影区', '南太阳阵1电压', '南太阳阵2电压',
                         '南太阳阵5电压', '南太阳阵6电压',
                         '南太阳阵7电压', '工况', '南太阳阵电流']

    data_column_north = ['北太阳电池温度1', '北太阳电池阵温度2', '北太阳电池阵温度3',
                         '光照区/阴影区', '北太阳阵1电压', '北太阳阵2电压',
                         '北太阳阵5电压', '北太阳阵6电压',
                         '北太阳阵7电压',  '工况', '北太阳阵电流']

    for file in uniform_files:
        df = pd.read_csv(file)
        data_south = df[data_column_south]
        X_south = data_south.drop(columns=['工况', '南太阳阵电流']).values
        label_south = data_south['工况'].values
        target_south = data_south['南太阳阵电流'].values

        data_north = df[data_column_north]
        X_north = data_north.drop(columns=['工况', '北太阳阵电流']).values
        label_north = data_north['工况'].values
        target_north = data_north['北太阳阵电流'].values

        X = np.concatenate((X_south, X_north), axis=0)
        label = np.concatenate((label_south, label_north), axis=0)
        target = np.concatenate((target_south, target_north), axis=0)

        all_data.append((X, label, target))

    data_features = np.vstack([data[0] for data in all_data])
    data_labels = np.hstack([data[1] for data in all_data])
    data_targets = np.hstack([data[2] for data in all_data])

    sample_X = data_features[::n]
    sample_label = data_labels[::n]
    sample_target = data_targets[::n]

    X_train, label_train, target_train, X_test, label_test, target_test = split_train_test(sample_X, sample_label, sample_target)

    return X_train, label_train, target_train, X_test, label_test, target_test


def inverse_y(y_out, timesteps, horizon, raw):
    n_timesteps = y_out.shape[0]
    n = n_timesteps + timesteps + horizon - 1
    y_restored = np.zeros(n)
    y_restored[timesteps + horizon - 1:] = y_out.flatten()
    y_restored[:timesteps + horizon - 1] = raw
    return y_restored


def build_classify_loaders(X_train, y_train, X_test, y_test, target_train, target_test,
                           time_steps, horizon, batch_size):

    train_data = np.concatenate((X_train, target_train.reshape(-1,1)), axis=1)
    test_data  = np.concatenate((X_test,  target_test.reshape(-1,1)), axis=1)
    std_train, std_test, _, _ = data_standarlization(train_data, test_data)


    std_train_feat = np.delete(std_train, -1, axis=1)
    std_test_feat  = np.delete(std_test,  -1, axis=1)

    Xtr_seq, ytr_seq = data_build(std_train_feat, y_train, time_steps, horizon, stride)
    Xte_seq, yte_seq = data_build(std_test_feat,  y_test,  time_steps, horizon, stride)


    lag_tr = TimeSeriesDataset(std_train, time_steps, horizon, stride).features.unsqueeze(-1)
    lag_te = TimeSeriesDataset(std_test,  time_steps, horizon, stride).features.unsqueeze(-1)

    Xtr = torch.tensor(Xtr_seq, dtype=torch.float32)
    Xte = torch.tensor(Xte_seq, dtype=torch.float32)
    ytr = torch.tensor(ytr_seq)
    yte = torch.tensor(yte_seq)

    Xtr_combin = torch.cat([Xtr, lag_tr], dim=2)
    Xte_combin = torch.cat([Xte, lag_te], dim=2)

    train_ds = TensorDataset(Xtr_combin, ytr[:,0])
    test_ds  = TensorDataset(Xte_combin, yte[:,0])

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False),
        std_train, std_test
    )


def build_regress_loaders_per_class(X_train, y_train, t_train,
                                    X_test,  y_test,  t_test,
                                    time_steps, horizon, batch_size, cls_idx):

    tr_mask = (y_train == cls_idx)
    te_mask = (y_test  == cls_idx)
    Xtr_c = X_train[tr_mask]
    ttr_c = t_train[tr_mask]
    Xte_c = X_test[te_mask]
    tte_c = t_test[te_mask]

    train_c = np.concatenate((Xtr_c, ttr_c.reshape(-1,1)), axis=1)
    test_c  = np.concatenate((Xte_c,  tte_c.reshape(-1,1)), axis=1)

    std_tr, std_te, m, s = data_standarlization(train_c, test_c)
    ds_tr = TimeSeriesDataset(std_tr, time_steps, horizon, stride)
    ds_te = TimeSeriesDataset(std_te, time_steps, horizon, stride)

    return (
        DataLoader(ds_tr, batch_size=batch_size, shuffle=True),
        DataLoader(ds_te, batch_size=batch_size, shuffle=False),
        m, s,
        len(Xtr_c), len(Xte_c)
    )


if __name__ == "__main__":
    # ----------------- 数据与超参 -----------------
    rout_files = ['data/Y2101_p.csv','data/Y2201_p.csv','data/Y2202_p.csv',
                  'data/Y2301_p.csv','data/Y2302_p.csv','data/Y2401_p.csv']
    mane_files = ['data/Y2203_p.csv','data/Y2402_p.csv']

    uniform_files = ['data/Y2101_p.csv', 'data/Y2201_p.csv', 'data/Y2202_p.csv',
                     'data/Y2203_p.csv',  'data/Y2301_p.csv', 'data/Y2302_p.csv',
                     'data/Y2401_p.csv', 'data/Y2402_p.csv']

    X_train, y_train, t_train, X_test, y_test, t_test = load_downsample(rout_files, mane_files)
    # X_train, y_train, t_train, X_test, y_test, t_test = uniform_downsample(uniform_files, n=90)

    batch_size = 64
    num_epochs_cls = 80
    num_epochs_reg = 100
    time_steps = 20
    horizon = 10
    stride = 10
    learning_rate = 1e-3

    input_channels = 10
    tcn_channels = [64,64,64,64,64,64,64]
    kernel_size = 3
    dropout = 0.10
    num_classes = 5
    alpha = 0.2

    hidden_dim = 256
    layers_num = 3
    attn_dim = 128
    input_dim = 1
    dropout_enc = 0.10
    increase_rate = 0.01
    decrease_rate = 0.01


    hyperparameters = {
        'lg': {'layers_num': 1, 'dropout_enc': 0.0, 'decay_coeff': 0.97},  # Sunlit region
        'es': {'layers_num': 3, 'dropout_enc': 0.1, 'decay_coeff': 0.92},  # Umbra
        'is': {'layers_num': 1, 'dropout_enc': 0.0, 'decay_coeff': 0.95},  # In-penumbra
        'os': {'layers_num': 5, 'dropout_enc': 0.1, 'decay_coeff': 0.98},  # Out-penumbra
        'sk': {'layers_num': 5, 'dropout_enc': 0.1, 'decay_coeff': 0.95},  # Orbit-keeping
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ----------------- 5 次独立重复（训练→测试）并取平均 -----------------
    seeds = [2025, 2026, 2027, 2028, 2029]

    # 指标累积
    acc_list = []
    mae_list = []
    mse_list = []
    rmse_list = []
    r2_list = []
    lat_cls_list = []
    lat_reg_list = []

    flops_cls = estimate_flops_tcn(input_channels, tcn_channels, kernel_size, time_steps)
    vram_cls = estimate_vram_inference_bytes(B=batch_size, T=time_steps, H=horizon, h=hidden_dim, dtype_bytes=4)

    params_reg = None
    flops_reg = None
    vram_reg = None

    # 训练延迟、推理延迟
    lat_cls_total = 0
    lat_reg_total = 0
    lat_steps_cls = 0
    lat_steps_reg = 0

    for run_id, seed in enumerate(seeds, 1):
        print(f"\n========== Run {run_id}/5 | Seed={seed} ==========")
        set_seed(seed)

        # --------- 分类：训练 → 测试 ---------
        train_loader_cls, test_loader_cls, std_tr_all, std_te_all = build_classify_loaders(
            X_train, y_train, X_test, y_test, t_train, t_test, time_steps, horizon, batch_size)
        model_cls = DTCN(input_channels, tcn_channels, kernel_size, dropout, num_classes, horizon).to(device)
        params_cls = count_trainable_parameters(model_cls) # 参数量

        criterion_cls = nn.CrossEntropyLoss()
        optim_cls = torch.optim.Adam(model_cls.parameters(), lr=learning_rate)
        sched_cls = optim.lr_scheduler.StepLR(optim_cls, step_size=50, gamma=0.1)


        # ======== 运行时/估算参数========
        USE_AMP = False
        OPT_KIND = "adam"
        GRAD_CKPT = False
        CUDA_CONTEXT_MB = 500
        CUDNN_WS_MB = 384
        ALLOC_HEADROOM = 0.18

        prefetch = 2
        F_cls = input_channels + 1
        pin_mem_mb_est = (batch_size * time_steps * F_cls * 4 * prefetch) / (1024 * 1024)

        if device.type == 'cuda':
            xb_cls, yb_cls = next(iter(build_classify_loaders(
                X_train, y_train, X_test, y_test, t_train, t_test,
                time_steps, horizon, batch_size
            )[0]))
            xb_cls = xb_cls.to(device)
            yb_cls = yb_cls.to(device)

            def cls_loss_fn(model, x, y):
                out = model(x)
                per_class, _ = aggregate_over_time(out, 'ordered1', alpha=alpha)
                return nn.CrossEntropyLoss()(per_class, y.long())

            # 训练显存实测
            probe_cls_train = measure_vram_peak_generic(
                model_cls, xb_cls, yb_cls, train=True, use_amp=USE_AMP,
                optimizer_name=OPT_KIND, lr=learning_rate, classify_loss_fn=cls_loss_fn)
            # 推理显存实测
            probe_cls_infer = measure_vram_peak_generic(
                model_cls, xb_cls, None, train=False, use_amp=USE_AMP, classify_loss_fn=cls_loss_fn)
        else:
            probe_cls_train = {"peak_reserved": 0, "peak_allocated": 0, "delta_allocated": 0}
            probe_cls_infer = {"peak_reserved": 0, "peak_allocated": 0, "delta_allocated": 0}

        est_reg_bytes_per_cond, _ = estimate_vram_bytes_full(
            AttentionRecursive(input_dim, hidden_dim, horizon, layers_num, dropout_enc, attn_dim),
            B=batch_size, T=time_steps, H=horizon, m=input_dim, h=hidden_dim, L=layers_num, attn_dim=attn_dim,
            mode="infer", dtype="fp32", optimizer=OPT_KIND, amp=USE_AMP, grad_ckpt=GRAD_CKPT,
            cuda_context_mb=CUDA_CONTEXT_MB, cudnn_ws_mb=CUDNN_WS_MB,
            allocator_headroom=ALLOC_HEADROOM, pin_mem_mb=pin_mem_mb_est, buffers_mb=64
        )

        # 模型训练
        avg_cls_latency = train_model_cls(train_loader_cls, model_cls, num_epochs_cls, criterion_cls, optim_cls, sched_cls,
                            device, alpha=alpha)

        # 测试
        y_pred, y_true, ms_per_step_cls = predict_rolling_classify(
            model_cls, test_loader_cls, criterion_cls, device,
            alpha=alpha, return_latency=True
        )
        # 分类模型延迟
        lat_cls_total += ms_per_step_cls
        lat_steps_cls += len(y_pred)

        acc = (y_pred == y_true).mean()
        acc_list.append(acc)

        if ms_per_step_cls is not None:
            lat_cls_list.append(ms_per_step_cls)
        print(f"[Perception] Acc={acc:.4f} | latency={ms_per_step_cls:.4f} ms/step")

        X_test_predClass = [[] for _ in range(num_classes)]
        t_test_predClass = [[] for _ in range(num_classes)]

        for i, cls in enumerate(y_pred):
            X_test_predClass[int(cls)].append(X_test[i])
            t_test_predClass[int(cls)].append(t_test[i])
        X_test_predClass = [np.stack(xs, axis=0) if len(xs)>0 else np.empty((0, X_test.shape[1])) for xs in X_test_predClass]
        t_test_predClass = [np.array(ts) if len(ts)>0 else np.empty((0,)) for ts in t_test_predClass]

        preds_all, trues_all = [], []
        lat_reg_accu, steps_reg_accu = 0.0, 0

        params_reg_total = 0
        flops_reg_total = 0
        vram_reg_total = 0
        avg_reg_latencies = 0

        vram_infer_reserved_list = []
        vram_train_reserved_list = []


        for cls_idx, tag in enumerate(['lg','es','is','os','sk']):
            dl_tr, dl_te, mu, sd, ntr, nte = build_regress_loaders_per_class(
                X_train, y_train, t_train, X_test, y_test, t_test,
                time_steps, horizon, batch_size, cls_idx
            )
            if nte == 0:
                continue

            params = hyperparameters[tag]

            model_reg = AttentionRecursive(
                input_dim,
                hidden_dim,
                horizon,
                params['layers_num'],
                dropout=params['dropout_enc'],
                attn_dim=attn_dim
            ).to(device)

            criterion_reg = nn.MSELoss()
            optim_reg = torch.optim.Adam(model_reg.parameters(), lr=learning_rate)
            sched_reg = optim.lr_scheduler.StepLR(optim_reg, step_size=50, gamma=0.1)


            if device.type == 'cuda':
                xb_te, yb_te = next(iter(dl_te))
                xb_tr, yb_tr = next(iter(dl_tr))
                xb_te = xb_te.unsqueeze(-1).to(device)
                yb_te = yb_te.to(device)
                xb_tr = xb_tr.unsqueeze(-1).to(device)
                yb_tr = yb_tr.to(device)

                probe_reg_train = measure_vram_peak_generic(
                    model_reg, xb_tr, yb_tr, train=True, use_amp=USE_AMP,
                    optimizer_name=OPT_KIND, lr=learning_rate, classify_loss_fn=None)

                probe_reg_infer = measure_vram_peak_generic(
                    model_reg, xb_te, None, train=False, use_amp=USE_AMP, classify_loss_fn=None)

                vram_train_reserved_list.append(probe_reg_train["peak_reserved"])
                vram_infer_reserved_list.append(probe_reg_infer["peak_reserved"])
            else:
                vram_train_reserved_list.append(0)
                vram_infer_reserved_list.append(0)


            # 模型训练
            avg_reg_latency = train_model_reg(dl_tr, model_reg, criterion_reg, optim_reg, sched_reg,
                                              device, num_epochs_reg, params['decay_coeff'],
                                              increase_rate, decrease_rate)

            # 测试
            pred_seq, true_seq, ms_per_step_reg = predict_rolling_window(model_reg, dl_te, mu, sd, device, return_latency=True)

            lat_reg_total += ms_per_step_reg
            lat_steps_reg += len(y_pred)

            preds_all.append(pred_seq)
            trues_all.append(true_seq)

            if ms_per_step_reg is not None:
                lat_reg_accu += ms_per_step_reg * len(pred_seq)
                steps_reg_accu += len(pred_seq)

            # 参数量
            params_reg = count_trainable_parameters(model_reg)
            params_reg_total += params_reg

            # FLOPs
            flops_reg = estimate_flops_attention_recursive(T=time_steps, H=horizon, m=input_dim, h=hidden_dim,
                                                           L=hyperparameters[tag]['layers_num'], attn_dim=attn_dim)
            flops_reg_total += flops_reg

            # VRAM
            vram_reg = estimate_vram_inference_bytes(B=batch_size, T=time_steps, H=horizon, h=hidden_dim, dtype_bytes=4) + params_reg*4
            vram_reg_total += vram_reg


        total_vram_infer_bytes = probe_cls_infer["peak_reserved"] + int(np.sum(vram_infer_reserved_list))
        total_vram_train_bytes = probe_cls_train["peak_reserved"] + int(np.sum(vram_train_reserved_list))

        print("\n================ VRAM ================")
        print(
            f"[分类] 推理峰值 ≈ {probe_cls_infer['peak_reserved'] / 1024 / 1024:.1f} MB | 训练峰值 ≈ {probe_cls_train['peak_reserved'] / 1024 / 1024:.1f} MB")
        for tag, b_infer, b_train in zip(['lg', 'es', 'is', 'os', 'sk'], vram_infer_reserved_list,
                                         vram_train_reserved_list):
            print(f"[{tag}] 推理峰值 ≈ {b_infer / 1024 / 1024:.1f} MB | 训练峰值 ≈ {b_train / 1024 / 1024:.1f} MB")
        print(f"[VRAM] 推理合计（分类+5工况）≈ {total_vram_infer_bytes / 1024 / 1024:.1f} MB")
        print(f"[VRAM] 训练合计（分类+5工况）≈ {total_vram_train_bytes / 1024 / 1024:.1f} MB")


        total_params = params_cls + params_reg_total
        print(f"可训练总参数量（分类器 + 5个回归模型）：{total_params:,}")

        total_flops = flops_cls + flops_reg_total
        print(f"总FLOPs（分类器 + 5个回归模型）：{total_flops:,} FLOPs")
        print(f"平均每个 epoch 工况识别的训练延迟：{avg_cls_latency:.4f} ms")
        print(f"平均每个 epoch 多个工况预测的训练延迟和：{avg_reg_latencies:.4f} ms")

        if len(preds_all)==0:
            raise RuntimeError("No regression predictions produced. Check class splits.")

        pred_concat = np.concatenate(preds_all, axis=0)
        true_concat = np.concatenate(trues_all, axis=0)

        mae = mean_absolute_error(true_concat, pred_concat)
        mse = mean_squared_error(true_concat, pred_concat)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_concat, pred_concat)

        mae_list.append(mae)
        mse_list.append(mse)
        rmse_list.append(rmse)
        r2_list.append(r2)

        lat_reg_mean = (lat_reg_accu/steps_reg_accu) if steps_reg_accu>0 else None

        if lat_reg_mean is not None:
            lat_reg_list.append(lat_reg_mean)
        print(f"[Regress] MAE={mae:.6f} | MSE={mse:.6f} |RMSE={rmse:.6f} | R2={r2:.4f} | latency={lat_reg_mean:.4f} ms/step")

    # ----------------- 平均结果 -----------------
    def avg(x):
        return float(np.mean(x)) if len(x)>0 else float('nan')

    print("\n================ AVERAGED OVER 5 RUNS ================")
    print(f"Classify Acc: {avg(acc_list):.4f}")
    print(f"Regress  MAE: {avg(mae_list):.6f} | MSE: {avg(mse_list):.6f} |RMSE: {avg(rmse_list):.6f} | R2: {avg(r2_list):.4f}")
    print(f"Latency  (ms/step): Classify {avg(lat_cls_list):.4f} | Regress {avg(lat_reg_list):.4f}")
    print(f"Throughput  (num/s): Classify {1000/ (avg(lat_cls_list)):.4f} | Regress {1000/ (avg(lat_reg_list)):.4f}")


    print("\n================ MODEL STATS (single model) ================")
    print(f"Classifier TCN FLOPs (approx, T={time_steps}): {flops_cls:,}")
    print(f"Classifier TCN Params (approx, T={time_steps}): {params_cls:,}")

    if params_reg is not None:
        print(f"Regressor Params: {params_reg:,}")
        print(f"Regressor FLOPs  (approx per sample): {flops_reg:,}")