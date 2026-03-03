"""
Fine-tuning script for GraphRiverCast (GRC)

This script implements the "pretrain-finetune" paradigm for regional adaptation.
Fine-tuning synergizes global hydrodynamic knowledge with sparse local observations.

Reference:
    Ren et al. "Global River Forecasting with a Topology-Informed AI Foundation Model"
"""
import os
import re
import argparse
import datetime as dt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import GraphRiverCast, GCN_GRU  # GCN_GRU is alias for backward compatibility


# ============================================================
# Freeze Profile Strategies
# ============================================================
FREEZE_PROFILES = {
    "p0_head": "Only readout layer (minimal fine-tuning)",
    "p1_head_norm": "Readout + normalization layers",
    "p2_spatial_last": "+ last GNN layer",
    "p3_temporal_input_only": "+ GRU input gates only",
    "p4_add_featmix": "+ feature mixing layers (recommended)",
    "p5_add_embed": "+ embedding layer",
    "p6_spatial_all": "+ all GNN layers",
    "p7_temporal_recurrent": "+ GRU recurrent gates",
    "p8_full": "All layers unfrozen",
    "p9_scratch": "All layers unfrozen (train from scratch)",
}


def get_profile_spec(model, profile_name):
    """
    Define freeze/unfreeze rules for different profiles.
    Returns a list of tuples: (regex_pattern, lr_multiplier, requires_grad)
    Last matching rule wins.
    """
    # Get model layer counts
    last_gnn_idx = getattr(model, "spatial_num_layer", 2) - 1
    last_gru_idx = getattr(model, "temporal_num_layer", 1) - 1

    # Base components
    readout = [(r"^readout\.(weight|bias)$", 1.00, True)]
    norms = [
        (r"^out_norm\.weight$", 0.30, True),
        (r"^fmix_norm\.weight$", 0.30, True),
        (r"^gnn_norm\.weight$", 0.30, True),
        (r"^gru_norm\.weight$", 0.30, True),
        (r"^hid_norm\.weight$", 0.30, True),
    ]

    # GNN components
    gnn_last = [(rf"^graph_encoder\.convs\.{last_gnn_idx}\..*(weight|bias)$", 0.50, True)]
    gnn_rest = [(r"^graph_encoder\.convs\.\d+\..*(weight|bias)$", 0.30, True)]

    # GRU components - input gates only
    gru_input_only = [
        (rf"^gru_cell\.weight_ih_l{last_gru_idx}$", 0.40, True),
        (rf"^gru_cell\.bias_ih_l{last_gru_idx}$", 0.40, True),
        (rf"^gru_cell\.weight_hh_l{last_gru_idx}$", 0.00, False),
        (rf"^gru_cell\.bias_hh_l{last_gru_idx}$", 0.00, False),
    ]
    # GRU components - recurrent gates
    gru_recurrent = [
        (rf"^gru_cell\.weight_hh_l{last_gru_idx}$", 0.20, True),
        (rf"^gru_cell\.bias_hh_l{last_gru_idx}$", 0.20, True),
    ]

    # Other components
    featmix = [
        (r"^feat_mix\.0\.(weight|bias)$", 0.50, True),
        (r"^feat_mix\.2\.(weight|bias)$", 0.50, True),
    ]
    embed = [(r"^embed\.(weight|bias)$", 0.20, True)]
    alpha_beta = [(r"^(alpha|beta)$", 0.10, True)]

    # Profile definitions (progressive unfreezing)
    profiles = {
        "p0_head": readout,
        "p1_head_norm": readout + norms + alpha_beta,
        "p2_spatial_last": readout + norms + alpha_beta + gnn_last,
        "p3_temporal_input_only": readout + norms + alpha_beta + gnn_last + gru_input_only,
        "p4_add_featmix": readout + norms + alpha_beta + gnn_last + gru_input_only + featmix,
        "p5_add_embed": readout + norms + alpha_beta + gnn_last + gru_input_only + featmix + embed,
        "p6_spatial_all": readout + norms + alpha_beta + gnn_rest + gnn_last + gru_input_only + featmix + embed,
        "p7_temporal_recurrent": readout + norms + alpha_beta + gnn_rest + gnn_last + gru_input_only + gru_recurrent + featmix + embed,
        "p8_full": [(r"^.*", 1.0, True)],
        "p9_scratch": [(r"^.*", 1.0, True)],
    }

    if profile_name not in profiles:
        raise ValueError(f"Unknown freeze profile: {profile_name}. Available: {list(profiles.keys())}")

    return profiles[profile_name]


def apply_freeze_profile(model, profile_name, base_lr, weight_decay=1e-4):
    """
    Apply freezing strategy and build parameter groups with different learning rates.

    Args:
        model: The model to apply freezing to
        profile_name: Name of the freezing profile
        base_lr: Base learning rate
        weight_decay: Weight decay value

    Returns:
        List of parameter groups for optimizer
    """
    spec = get_profile_spec(model, profile_name)

    # Step 1: Freeze all parameters by default
    for p in model.parameters():
        p.requires_grad = False

    # Step 2: Apply profile rules (last match wins)
    name2lr = {}
    for name, p in model.named_parameters():
        lr_mult, req_grad = None, None
        for pat, mult, req in spec:
            if re.search(pat, name):
                lr_mult, req_grad = mult, req
        if req_grad is not None:
            p.requires_grad = req_grad
            if req_grad:
                name2lr[name] = base_lr * (lr_mult if lr_mult else 1.0)

    # Step 3: Build parameter groups
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        lr = name2lr.get(name, base_lr * 0.05)  # Fallback LR

        # Exclude bias/norm/embed from weight decay
        lname = name.lower()
        no_wd = (
            (p.ndim <= 1) or
            ("norm" in lname) or
            (lname.endswith(".bias")) or
            (lname.startswith("embed.")) or
            (lname in ["alpha", "beta"])
        )
        wd = 0.0 if no_wd else weight_decay

        key = f"{lr}_{wd}"
        if key not in groups:
            groups[key] = {"params": [], "lr": lr, "weight_decay": wd, "names": []}
        groups[key]["params"].append(p)
        groups[key]["names"].append(name)

    param_groups = list(groups.values())

    # Print statistics
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Freeze profile: {profile_name}")
    print(f"[INFO] Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    print(f"[INFO] Parameter groups: {len(param_groups)}")

    return param_groups


# ============================================================
# Dataset
# ============================================================
class FinetuneDataset(Dataset):
    """Dataset for fine-tuning"""

    def __init__(self, data_dict, hist_len, fut_len, is_train=True):
        self.data = data_dict['train' if is_train else 'val']
        self.obs = data_dict['obs_train' if is_train else 'obs_val']
        self.static_var = data_dict['static_var']
        self.edge_index = data_dict['edge_index']
        self.meanstd = data_dict['meanstd_dynamic']

        self.hist_len = hist_len
        self.fut_len = fut_len
        self.seq_len = hist_len + fut_len
        self.sample_len = self.data.shape[0] - self.seq_len + 1

    def __len__(self):
        return self.sample_len

    def __getitem__(self, idx):
        data_win = self.data[idx:idx + self.seq_len]
        obs_fut = self.obs[idx + self.hist_len:idx + self.seq_len]

        return {
            'meanstd': self.meanstd,
            'river_hist': data_win[:self.hist_len][..., :3],
            'river_fut': data_win[self.hist_len:][..., :3],
            'runoff_hist': data_win[:self.hist_len][..., -1:],
            'runoff_fut': data_win[self.hist_len:][..., -1:],
            'static_var': self.static_var,
            'edge_index': self.edge_index,
            'obs_values_fut': obs_fut,
        }


# ============================================================
# Utilities
# ============================================================
def days_index_2000(y, m, d):
    base = dt.date(2000, 1, 1)
    cur = dt.date(int(y), int(m), int(d))
    return (cur - base).days


def parse_date(date_str):
    dt_obj = dt.datetime.strptime(date_str, '%Y-%m-%d')
    return dt_obj.year, dt_obj.month, dt_obj.day


def calc_nse(pred, obs, mask):
    """Calculate NSE loss (1 - NSE for minimization)"""
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    obs_filled = torch.where(mask, obs, torch.zeros_like(obs))
    pred_filled = torch.where(mask, pred, torch.zeros_like(pred))

    node_cnt = mask.sum(dim=(0, 1))
    has_obs = node_cnt > 0
    node_mean = obs_filled.sum(dim=(0, 1)) / node_cnt.clamp_min(1.0)

    rss = (((pred_filled - obs_filled) ** 2) * mask).sum(dim=(0, 1))
    tss = ((((obs_filled - node_mean) * mask) ** 2)).sum(dim=(0, 1))

    valid_mask = (tss > 0.1) & has_obs
    if valid_mask.sum() > 0:
        nse_per_node = 1.0 - (rss[valid_mask] / tss[valid_mask])
        loss = (1.0 - nse_per_node).mean()
    else:
        loss = F.mse_loss(pred_filled[mask], obs_filled[mask])

    return loss


# ============================================================
# Data Preparation
# ============================================================
def prepare_finetune_data(data_dir, group, config):
    """Prepare data for fine-tuning"""
    folder = os.path.join(data_dir, group)

    # Load data
    dynamic_ds = np.load(os.path.join(folder, "dynamic_var.npz"))
    static_ds = np.load(os.path.join(folder, "static_var.npz"))
    edge_index = np.load(os.path.join(folder, "edge_index.npy"))
    finetune_ds = np.load(os.path.join(folder, "FineTuning.npz"))

    # Dynamic variables
    dynamic_vars = ['outflw', 'rivdph', 'storage', 'runoff']
    dynamic_all = np.stack([dynamic_ds[k] for k in dynamic_vars], axis=-1).astype(np.float32)

    # Static variables
    static_vars = ['ctmare', 'elevtn', 'grdare', 'nxtdst', 'rivlen',
                  'rivwth_gwdlr', 'uparea', 'width', 'fldhgt']
    static_arrays = [arr[:, np.newaxis] if arr.ndim == 1 else arr
                   for arr in [static_ds[k] for k in static_vars]]
    static_all = np.concatenate(static_arrays, axis=-1).astype(np.float32)

    # Observation data
    finetune_obs = finetune_ds['OBS'].astype(np.float32)
    finetune_node_idx = finetune_ds['node_idx']

    # Create full observation array
    full_obs = np.full((finetune_obs.shape[0], dynamic_all.shape[1]), np.nan, dtype=np.float32)
    full_obs[:, finetune_node_idx] = finetune_obs

    # Time indices
    pretrain_start = days_index_2000(*parse_date(config['pretrain_start']))
    pretrain_end = days_index_2000(*parse_date(config['pretrain_end']))
    train_start = days_index_2000(*parse_date(config['train_start']))
    train_end = days_index_2000(*parse_date(config['train_end']))
    val_start = days_index_2000(*parse_date(config['val_start']))
    val_end = days_index_2000(*parse_date(config['val_end']))

    # Split data
    pretrainset = dynamic_all[pretrain_start:pretrain_end + 1]
    trainset = dynamic_all[train_start:train_end + 1]
    valset = dynamic_all[val_start:val_end + 1]
    train_obs = full_obs[train_start:train_end + 1]
    val_obs = full_obs[val_start:val_end + 1]

    # Normalization (node_variable)
    meanstd_dynamic = {
        'mean': np.mean(pretrainset, axis=0),
        'std': np.std(pretrainset, axis=0),
    }
    meanstd_dynamic['std'][meanstd_dynamic['std'] == 0] = 1e-8

    trainset = (trainset - meanstd_dynamic['mean']) / meanstd_dynamic['std']
    valset = (valset - meanstd_dynamic['mean']) / meanstd_dynamic['std']
    train_obs = (train_obs - meanstd_dynamic['mean'][:, 0]) / meanstd_dynamic['std'][:, 0]
    val_obs = (val_obs - meanstd_dynamic['mean'][:, 0]) / meanstd_dynamic['std'][:, 0]

    # Static normalization
    meanstd_static = {
        'mean': np.mean(static_all, axis=0),
        'std': np.std(static_all, axis=0)
    }
    meanstd_static['std'][meanstd_static['std'] == 0] = 1e-8
    static_norm = (static_all - meanstd_static['mean']) / meanstd_static['std']

    return {
        'train': trainset.astype(np.float32),
        'val': valset.astype(np.float32),
        'obs_train': train_obs.astype(np.float32),
        'obs_val': val_obs.astype(np.float32),
        'static_var': static_norm.astype(np.float32),
        'edge_index': edge_index,
        'meanstd_dynamic': meanstd_dynamic,
    }


# ============================================================
# Training Loop
# ============================================================
def train_one_epoch(model, dataloader, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    # Create progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]", leave=True)

    for batch in pbar:
        # Move to device
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        # Handle meanstd dict
        if isinstance(inputs.get('meanstd'), dict):
            inputs['meanstd'] = {
                k: torch.from_numpy(v).to(device).float() if isinstance(v, np.ndarray) else v
                for k, v in inputs['meanstd'].items()
            }

        optimizer.zero_grad()

        outputs = model(inputs)

        # Calculate loss
        pred = outputs['river_fut_hat'][..., 0]  # [B, T, N]
        obs = inputs['obs_values_fut']
        mask = ~torch.isnan(obs)

        loss = calc_nse(pred, obs, mask)

        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss

        # Update progress bar with loss
        pbar.set_postfix({"Loss": f"{batch_loss:.4f}"})

    avg_loss = total_loss / num_batches
    avg_nse = 1.0 - avg_loss
    return avg_loss, avg_nse


def validate(model, dataloader, device):
    model.eval()

    # Accumulate predictions and observations across all batches
    all_preds = []
    all_obs = []

    # Create progress bar
    pbar = tqdm(dataloader, desc="[Validation]", leave=True)

    with torch.no_grad():
        for batch in pbar:
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            if isinstance(inputs.get('meanstd'), dict):
                inputs['meanstd'] = {
                    k: torch.from_numpy(v).to(device).float() if isinstance(v, np.ndarray) else v
                    for k, v in inputs['meanstd'].items()
                }

            outputs = model(inputs)

            pred = outputs['river_fut_hat'][..., 0]  # [B, T, N]
            obs = inputs['obs_values_fut']           # [B, T, N]

            all_preds.append(pred.cpu())
            all_obs.append(obs.cpu())

    # Concatenate all batches: [total_samples, T, N]
    all_preds = torch.cat(all_preds, dim=0)
    all_obs = torch.cat(all_obs, dim=0)
    all_mask = ~torch.isnan(all_obs)

    # Compute NSE once on the full validation set
    loss = calc_nse(all_preds, all_obs, all_mask)
    avg_loss = loss.item()
    avg_nse = 1.0 - avg_loss

    return avg_loss, avg_nse


# ============================================================
# Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune GraphRiverCast (GRC) Model")

    p.add_argument("--ckpt", type=str, default=None, help="Pretrained checkpoint (None for scratch)")
    p.add_argument("--data-dir", type=str, default="data", help="Data directory")
    p.add_argument("--group", type=str, default="LamaH_CE06min_obs2000_2017", help="Dataset group")
    p.add_argument("--save-dir", type=str, default="results/finetune", help="Output directory")

    # Time settings
    p.add_argument("--pretrain-start", default="2000-01-01", help="Normalization start")
    p.add_argument("--pretrain-end", default="2017-12-31", help="Normalization end")
    p.add_argument("--train-start", default="2000-01-01", help="Training start")
    p.add_argument("--train-end", default="2009-12-31", help="Training end")
    p.add_argument("--val-start", default="2010-01-01", help="Validation start")
    p.add_argument("--val-end", default="2017-12-31", help="Validation end")

    # Training settings
    p.add_argument("--hist", default=14, type=int, help="History window")
    p.add_argument("--future", default=14, type=int, help="Future window")
    p.add_argument("--batch-size", default=16, type=int, help="Batch size")
    p.add_argument("--epochs", default=50, type=int, help="Number of epochs")
    p.add_argument("--lr", default=1e-4, type=float, help="Base learning rate")
    p.add_argument("--weight-decay", default=1e-4, type=float, help="Weight decay")
    p.add_argument("--device", default="cpu", choices=["cuda", "cpu", "mps"], help="Device")

    # Freeze profile
    p.add_argument(
        "--freeze-profile",
        default="p4_add_featmix",
        choices=list(FREEZE_PROFILES.keys()),
        help="Layer freezing strategy for fine-tuning. Options:\n" +
             "\n".join([f"  {k}: {v}" for k, v in FREEZE_PROFILES.items()])
    )

    return p.parse_args()


def main():
    args = parse_args()

    # Setup device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Checkpoint: {args.ckpt or 'None (training from scratch)'}")

    # Prepare data
    config = {
        'pretrain_start': args.pretrain_start,
        'pretrain_end': args.pretrain_end,
        'train_start': args.train_start,
        'train_end': args.train_end,
        'val_start': args.val_start,
        'val_end': args.val_end,
    }
    data_dict = prepare_finetune_data(args.data_dir, args.group, config)

    # Create datasets
    train_ds = FinetuneDataset(data_dict, args.hist, args.future, is_train=True)
    val_ds = FinetuneDataset(data_dict, args.hist, args.future, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"[INFO] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Create model
    task = {
        "type": "finetune",
        "window": {"finetune": {"history": args.hist, "future": args.future}},
    }

    if args.ckpt:
        # Load pretrained checkpoint
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        try:
            cfg = ckpt["hyper_parameters"]["model_arch"]["GCN_GRU"]["cfg"]
        except KeyError:
            cfg = GCN_GRU.DEFAULT_CONFIG.copy()

        model = GCN_GRU(cfg, task)

        # Load weights (with key renaming for backward compatibility)
        state_dict = ckpt.get("state_dict", ckpt)
        new_state = {}
        for k, v in state_dict.items():
            key = k[4:] if k.startswith("net.") else k
            key = key.replace("gcn.", "graph_encoder.")  # Rename for backward compatibility
            new_state[key] = v
        model.load_state_dict(new_state, strict=False)
        print("[INFO] Loaded pretrained weights")
    else:
        model = GCN_GRU(GCN_GRU.DEFAULT_CONFIG.copy(), task)
        print("[INFO] Training from scratch")

    model.to(device)

    # Apply freeze profile and create optimizer
    if args.freeze_profile == "p9_scratch" or args.ckpt is None:
        # Training from scratch: no freezing, use all parameters
        print("[INFO] Training from scratch - all parameters trainable")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        # Fine-tuning: apply freeze profile
        param_groups = apply_freeze_profile(
            model,
            args.freeze_profile,
            base_lr=args.lr,
            weight_decay=args.weight_decay
        )
        optimizer = torch.optim.Adam(param_groups)

    # Training loop
    best_val_loss = float('inf')
    best_val_nse = float('-inf')

    print(f"\n{'='*60}")
    print(f"Starting Training: {args.epochs} epochs, {len(train_loader)} batches/epoch")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        # Training
        train_loss, train_nse = train_one_epoch(model, train_loader, optimizer, device, epoch, args.epochs)

        # Validation
        val_loss, val_nse = validate(model, val_loader, device)

        # Epoch summary
        print(f"\n[Epoch {epoch + 1} Summary]")
        print(f"  Train: Loss={train_loss:.4f}, NSE={train_nse:.4f}")
        print(f"  Val:   Loss={val_loss:.4f}, NSE={val_nse:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_nse = val_nse
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_nse': val_nse,
            }, os.path.join(args.save_dir, "best_model.pt"))
            print(f"  -> Saved best model (val_loss={val_loss:.4f}, val_nse={val_nse:.4f})")

    print(f"\n{'='*60}")
    print(f"[OK] Training completed!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best val NSE:  {best_val_nse:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
