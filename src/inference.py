"""
Inference script for GraphRiverCast (GRC)

This script runs GRC-HotStart mode inference using pre-trained checkpoints.
GRC-HotStart initializes with river states for maximum short-term forecasting fidelity.

Reference:
    Ren et al. "Global River Forecasting with a Topology-Informed AI Foundation Model"
"""
import os
import json
import csv
import time
import argparse
import datetime as dt
import numpy as np
import torch

from model import GraphRiverCast, GCN_GRU  # GCN_GRU is alias for backward compatibility


def parse_args():
    p = argparse.ArgumentParser(description="GraphRiverCast Inference (GRC-HotStart Mode)")

    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
    p.add_argument("--data-dir", type=str, default="data", help="Data directory")
    p.add_argument("--group", type=str, default="LamaH_CE06min_obs2000_2017", help="Dataset group name")
    p.add_argument("--save-dir", type=str, default="results", help="Output directory")

    # Time window settings
    p.add_argument("--start", default="2009-01-01", type=str, help="Prediction start date")
    p.add_argument("--hist", default=365, type=int, help="History window length (days)")
    p.add_argument("--future", default=2922, type=int, help="Future window length (days)")
    p.add_argument("--fit-start", default="2000-01-01", type=str, help="Normalization fit start")
    p.add_argument("--fit-end", default="2017-12-31", type=str, help="Normalization fit end")

    p.add_argument("--device", default="cpu", choices=["cuda", "cpu", "mps"], help="Device")

    return p.parse_args()


def ymd_tuple(s: str):
    y, m, d = map(int, s.split("-"))
    return (y, m, d)


def add_days(s: str, days: int) -> str:
    y, m, d = map(int, s.split("-"))
    return (dt.date(y, m, d) + dt.timedelta(days=days)).strftime("%Y-%m-%d")


def days_index_2000(y, m, d) -> int:
    base = dt.date(2000, 1, 1)
    cur = dt.date(int(y), int(m), int(d))
    return (cur - base).days


def load_checkpoint(ckpt_path: str, device: torch.device):
    """Load checkpoint and extract model config and state dict"""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Extract config
    try:
        cfg = checkpoint["hyper_parameters"]["model_arch"]["GCN_GRU"]["cfg"]
    except KeyError:
        # Use default config if not found
        cfg = GCN_GRU.DEFAULT_CONFIG.copy()

    if "spatial_num_layer" not in cfg:
        cfg["spatial_num_layer"] = 2

    # Clean state dict (remove 'net.' prefix and rename gcn->graph_encoder)
    raw_state = checkpoint.get("state_dict", checkpoint)
    new_state = {}
    for k, v in raw_state.items():
        # Remove 'net.' prefix if present
        key = k[4:] if k.startswith("net.") else k
        # Rename gcn to graph_encoder for backward compatibility
        key = key.replace("gcn.", "graph_encoder.")
        new_state[key] = v

    return cfg, new_state


def prepare_data(data_dir, group, start, hist, future, fit_start, fit_end):
    """Prepare data for inference"""
    folder = os.path.join(data_dir, group)

    # Load data files
    dynamic_ds = np.load(os.path.join(folder, "dynamic_var.npz"))
    static_ds = np.load(os.path.join(folder, "static_var.npz"))
    edge_index = np.load(os.path.join(folder, "edge_index.npy"))

    # Dynamic variables
    dynamic_variables = ['outflw', 'rivdph', 'storage', 'runoff']
    dynamic_all = np.stack([dynamic_ds[k] for k in dynamic_variables], axis=-1)

    # Static variables
    static_variables = ['ctmare', 'elevtn', 'grdare', 'nxtdst', 'rivlen',
                       'rivwth_gwdlr', 'uparea', 'width', 'fldhgt']
    static_arrays = [arr[:, np.newaxis] if arr.ndim == 1 else arr
                    for arr in [static_ds[k] for k in static_variables]]
    static_all = np.concatenate(static_arrays, axis=-1)

    # Time indices
    fit_s = days_index_2000(*ymd_tuple(fit_start))
    fit_e = days_index_2000(*ymd_tuple(fit_end))
    seq_len = hist + future
    pred_s = days_index_2000(*ymd_tuple(start))
    pred_e = pred_s + seq_len - 1

    # Slices
    fitset = dynamic_all[fit_s:fit_e + 1]
    predictset = dynamic_all[pred_s:pred_e + 1]

    # Normalization (node_variable)
    d_mean = np.mean(fitset, axis=0)
    d_std = np.std(fitset, axis=0)
    d_std[d_std == 0] = 1e-8
    predict_norm = (predictset - d_mean) / d_std

    s_mean = np.mean(static_all, axis=0)
    s_std = np.std(static_all, axis=0)
    s_std[s_std == 0] = 1e-8
    static_norm = (static_all - s_mean) / s_std

    # Split history and future
    river_ch = 3
    hist_arr = predict_norm[:hist]
    fut_arr = predict_norm[hist:]

    return {
        'river_hist': hist_arr[..., :river_ch],
        'river_fut': fut_arr[..., :river_ch],
        'runoff_hist': hist_arr[..., -1, np.newaxis],
        'runoff_fut': fut_arr[..., -1, np.newaxis],
        'static_var': static_norm,
        'edge_index': edge_index,
        'meanstd_dynamic': {'mean': d_mean, 'std': d_std},
        'meta': {
            'dynamic_variables': dynamic_variables,
            'static_variables': static_variables,
            'fit_range': {'start': fit_start, 'end': fit_end},
            'pred_range': {'start': start, 'end': add_days(start, seq_len - 1)},
        }
    }


def main():
    args = parse_args()

    # Setup device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Create unique output folder with timestamp
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
    run_name = f"{args.group}_{ckpt_name}_{timestamp}"
    output_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Checkpoint: {args.ckpt}")
    print(f"[INFO] Data: {args.data_dir}/{args.group}")
    print(f"[INFO] Output: {output_dir}")

    # ===============================
    # Phase 1: Load model and data
    # ===============================
    t_start = time.perf_counter()

    cfg, state_dict = load_checkpoint(args.ckpt, device)

    task = {
        "type": "predict",
        "window": {"predict": {"history": args.hist, "future": args.future}},
    }
    model = GCN_GRU(cfg, task)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")
    model.to(device).eval()

    # Prepare data
    data = prepare_data(
        args.data_dir, args.group, args.start,
        args.hist, args.future, args.fit_start, args.fit_end
    )

    # Build batch
    to_t = lambda x: torch.from_numpy(x).to(device).float()
    batch = {
        "river_hist": to_t(data['river_hist']).unsqueeze(0),
        "river_fut": to_t(data['river_fut']).unsqueeze(0),
        "runoff_hist": to_t(data['runoff_hist']).unsqueeze(0),
        "runoff_fut": to_t(data['runoff_fut']).unsqueeze(0),
        "static_var": to_t(data['static_var']).unsqueeze(0),
        "edge_index": torch.from_numpy(data['edge_index']).to(device).long().unsqueeze(0),
    }

    meanstd = data['meanstd_dynamic']
    meanstd_tensor = {
        "mean": torch.from_numpy(meanstd["mean"]).to(device).float().unsqueeze(0),
        "std": torch.from_numpy(meanstd["std"]).to(device).float().unsqueeze(0),
    }
    batch["meanstd"] = meanstd_tensor

    t_data_end = time.perf_counter()

    # ===============================
    # Phase 2: Inference
    # ===============================
    t_infer_start = time.perf_counter()

    with torch.no_grad():
        outputs = model(batch)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    t_infer_end = time.perf_counter()

    # ===============================
    # Phase 3: Post-processing
    # ===============================
    t_post_start = time.perf_counter()

    river_ch = 3
    pred_std = meanstd_tensor["std"][:, :, :river_ch]
    pred_mean = meanstd_tensor["mean"][:, :, :river_ch]

    river_fut_hat = outputs["river_fut_hat"] * pred_std.unsqueeze(1) + pred_mean.unsqueeze(1)
    river_fut_gt = batch["river_fut"] * pred_std.unsqueeze(1) + pred_mean.unsqueeze(1)

    # To numpy
    pred_np = river_fut_hat.squeeze(0).cpu().numpy()
    gt_np = river_fut_gt.squeeze(0).cpu().numpy()

    # Save results
    np.save(os.path.join(output_dir, "prediction.npy"), pred_np)
    np.save(os.path.join(output_dir, "groundtruth.npy"), gt_np)

    t_post_end = time.perf_counter()

    # ===============================
    # Save metadata and timing
    # ===============================
    duration_data = t_data_end - t_start
    duration_infer = t_infer_end - t_infer_start
    duration_post = t_post_end - t_post_start
    total_time = duration_data + duration_infer + duration_post

    num_reaches = data['static_var'].shape[0]
    num_days = args.future
    efficiency = (duration_infer / (num_reaches * num_days)) * 1e6

    meta = {
        "ckpt": args.ckpt,
        "group": args.group,
        "start": args.start,
        "hist": args.hist,
        "future": args.future,
        "device": str(device),
        "shapes": {
            "pred": list(pred_np.shape),
            "gt": list(gt_np.shape),
        },
        "timing": {
            "data_prep_sec": duration_data,
            "inference_sec": duration_infer,
            "post_process_sec": duration_post,
            "total_sec": total_time,
            "efficiency_us_per_reach_day": efficiency
        }
    }

    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Inference completed.")
    print(f"     Prediction shape: {pred_np.shape}")
    print(f"     Time: Data={duration_data:.2f}s, Infer={duration_infer:.2f}s, Post={duration_post:.2f}s")
    print(f"     Efficiency: {efficiency:.4f} us/(reach*day)")
    print(f"     Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
