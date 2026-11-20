
import os
import time
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Subset

from trainer import *
from data import *
from plotter import Plotter


# --------------------------------------------------
# Configuration
# --------------------------------------------------

NUM_TOKENS = 9

# Features used by each detector type.
TOKEN_FEATURES = {
    0: [0, 1, 2],  # SVT layer 1: x, y, z
    1: [0, 1, 2],  # SVT layer 2
    2: [0, 1, 2],  # SVT layer 3

    3: [0, 1],     # BMTC layer 4: r, z
    4: [0, 1],     # BMTC layer 7
    5: [0, 1],     # BMTC layer 9

    6: [0, 1, 2],  # BMTZ layer 5: r, sin(phi), cos(phi)
    7: [0, 1, 2],  # BMTZ layer 6
    8: [0, 1, 2],  # BMTZ layer 8
}


# --------------------------------------------------
# Command-line arguments
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="CVT Transformer cross-combination binary classification")

    # Device selection
    parser.add_argument("--device", type=str, choices=["cpu", "gpu", "auto"], default="auto")

    # Input files
    parser.add_argument("inputs", type=str, nargs="*", default=["realFakeCrossCombos.csv"], help="One or more input CSV files")

    # Training parameters
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Transformer parameters
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)

    # Output
    parser.add_argument("--outdir", type=str, default="outputs/local")
    parser.add_argument("--end_name", type=str, default="")

    # Runtime options
    parser.add_argument("--no_train", action="store_true", help="Skip training and use a saved model")
    parser.add_argument("--enable_progress_bar", action="store_true")

    return parser.parse_args()


# --------------------------------------------------
# Random seed
# --------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)


# --------------------------------------------------
# Calculate feature normalization statistics
# --------------------------------------------------

def calculate_feature_stats(features, valid_mask):
    """
    Calculate normalization statistics independently
    for each detector layer and feature.

    Input:
        features:   [N, 9, 3]
        valid_mask: [N, 9]

    Output:
        mean: [9, 3]
        std:  [9, 3]

    Missing measurements are excluded.

    BMTC uses only r and z.
    Its third feature is assigned mean=0, std=1.
    """

    mean = np.zeros((NUM_TOKENS, 3), dtype=np.float32)
    std = np.ones((NUM_TOKENS, 3), dtype=np.float32)

    for token in range(NUM_TOKENS):

        # Select valid measurements for this layer.
        valid = valid_mask[:, token]

        if not np.any(valid):
            raise ValueError(f"No valid measurements found for token {token} (layer {LAYER_IDS[token]}).")

        for feature_idx in TOKEN_FEATURES[token]:
            values = features[valid, token, feature_idx]

            mean[token, feature_idx] = values.mean()
            std_value = values.std()

            # Avoid division by zero.
            std[token, feature_idx] = max(float(std_value), 1e-8)

    return {
        "mean": mean.tolist(),
        "std": std.tolist()
    }

# --------------------------------------------------
# Save normalization statistics
# --------------------------------------------------

def save_stats(feature_stats, feature_stats_path):
    with open(feature_stats_path, "w") as f:
        json.dump(feature_stats, f, indent=2)


# --------------------------------------------------
# Load normalization statistics
# --------------------------------------------------

def load_stats(feature_stats_path):
    with open(feature_stats_path, "r") as f:
        feature_stats = json.load(f)

    return feature_stats

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    set_seed(42)

    args = parse_args()

    inputs = args.inputs if args.inputs else ["sample.csv"]
    outDir = args.outdir
    os.makedirs(outDir, exist_ok=True)
    doTraining = not args.no_train
    end_name = args.end_name

    # Repository paths.
    BASE_DIR = Path(__file__).resolve().parent
    NETS_DIR = BASE_DIR / "nets"
    NETS_DIR.mkdir(parents=True, exist_ok=True)

    # Normalization files.
    feature_stats_out_path = (Path(outDir) / f"feature_stats.json")

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    print("\nLoading data...")

    startT_data = time.time()

    features_all = []
    masks_all = []
    labels_all = []

    for fname in inputs:
        print(f"Loading {fname} ...")

        features, valid_mask, labels = (read_combinations(fname))

        features_all.extend(features)
        masks_all.extend(valid_mask)
        labels_all.extend(labels)

    features_all = np.asarray(features_all, dtype=np.float32)

    masks_all = np.asarray(masks_all, dtype=bool)

    labels_all = np.asarray(labels_all, dtype=np.float32)

    if len(features_all) == 0:
        raise ValueError("No valid tracks were loaded.")

    print("Number of tracks:", len(features_all))
    print("Features shape:", features_all.shape)
    print("Mask shape:", masks_all.shape)
    print("labels shape:", labels_all.shape)

    # --------------------------------------------------
    # Train / validation split
    # --------------------------------------------------
    n_samples = len(features_all)
    if n_samples < 2:
        raise ValueError("At least two tracks are required.")

    generator = torch.Generator().manual_seed(42)

    indices = torch.randperm(n_samples, generator=generator).numpy()

    train_size = int(0.8 * n_samples)
    val_size = (len(features_all) - train_size)
    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:].tolist()
    if train_size == 0 or len(val_indices) == 0:
        raise ValueError("Invalid training/validation split.")

    print("\nTrain size:", len(train_indices))
    print("Validation size:", len(val_indices))

    # Calculate pos_weight using training samples only
    train_labels = labels_all[train_indices]
    num_pos = (train_labels == 1).sum().item()
    num_neg = (train_labels == 0).sum().item()
    if num_pos == 0:
        raise ValueError("No positive samples in training set.")
    pos_weight = num_neg / num_pos

    # --------------------------------------------------
    # Normalization statistics
    # --------------------------------------------------

    if doTraining:
        print("\nCalculating normalization statistics...")

        # Use training samples only.
        train_features = features_all[train_indices]
        train_masks = masks_all[train_indices]

        feature_stats = calculate_feature_stats(train_features, train_masks)
        save_stats(feature_stats, feature_stats_out_path)

        print("\nFeature statistics:")
        for token, layer in enumerate(LAYER_IDS):
            print(f"Layer {layer}: mean={feature_stats['mean'][token]}, std={feature_stats['std'][token]}")

        print("\nSaved normalization statistics.")

    else:
        print("\nLoading normalization statistics from nets/...")

        feature_stats_path = (NETS_DIR / f"feature_stats.json")

        if not feature_stats_path.exists():
             raise FileNotFoundError(f"Normalization statistics not found: {feature_stats_path}")

        feature_stats = load_stats(feature_stats_path)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------
    dataset = comboDataset(features=features_all, valid_mask=masks_all, labels=labels_all, normalize=True, feature_stats=feature_stats)

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    batch_size = args.batch_size

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Inspect one batch.
    features_sample, mask_sample, labels_sample = next(iter(train_loader))

    print("\nBatch inspection:")
    print("Features:", features_sample.shape)
    print("Valid mask:", mask_sample.shape)
    print("labels:", labels_sample.shape)

    endT_data = time.time()

    print(f"\nData preparation took {endT_data - startT_data:.2f} seconds.")

    # --------------------------------------------------
    # Plotting
    # --------------------------------------------------
    plotter = Plotter(print_dir=outDir, end_name=end_name)

    # --------------------------------------------------
    # Model initialization
    # --------------------------------------------------

    if args.hidden_dim % args.nhead != 0:
        raise ValueError("hidden_dim must be divisible by nhead.")

    model = comboClassifierTransformer(hidden_dim=args.hidden_dim, nhead=args.nhead, num_layers=args.num_layers, lr=args.lr, pos_weight=pos_weight)

    loss_tracker = LossTracker()

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    if doTraining:
        if args.device == "cpu":
            accelerator, devices = "cpu", 1
        elif args.device == "gpu":
            if torch.cuda.is_available():
                accelerator, devices = "gpu", 1
            else:
                print("GPU unavailable. Using CPU.")
                accelerator, devices = "cpu", 1
        else:
            if torch.cuda.is_available():
                accelerator, devices = "gpu", 1
            else:
                accelerator, devices = "cpu", 1

        print(f"\nUsing accelerator={accelerator}, devices={devices}")

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            strategy="auto",
            max_epochs=args.max_epochs,
            enable_progress_bar=args.enable_progress_bar,
            log_every_n_steps=100,
            enable_checkpointing=False,
            check_val_every_n_epoch=1,
            num_sanity_val_steps=0,
            logger=False,
            callbacks=[loss_tracker]
        )

        print("\nTraining Transformer...")

        startT_train = time.time()

        trainer.fit(model, train_loader, val_loader)

        endT_train = time.time()

        print(f"\nTraining took {(endT_train - startT_train) / 60:.2f} minutes.")

        # Plot training loss.
        plotter.plotTrainLoss(loss_tracker)

        # --------------------------------------------------
        # Export TorchScript
        # --------------------------------------------------

        print("\nExporting TorchScript model...")

        model = model.to("cpu")
        model.eval()

        wrapper_model = comboClassifierTransformerWrapper(model)
        wrapper_model.eval()

        model_file = (Path(outDir) / f"transformer_{end_name}.pt")

        torchscript_model = torch.jit.script(wrapper_model)
        torchscript_model.save(model_file)

        print("Saved model:", model_file)

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    else:
        model_file = (NETS_DIR / f"transformer_default.pt")

    model_file = model_file.resolve()

    print("\nLoading model from:", model_file)

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    model = torch.jit.load(str(model_file))
    model.eval()

    # --------------------------------------------------
    # Validation inference
    # --------------------------------------------------
    val_loader_test = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    all_probs = []
    all_labels = []

    print("\nRunning validation inference...")

    startT_test = time.time()

    with torch.no_grad():
        for features_batch, mask_batch, label_batch in val_loader_test:
            features_batch = features_batch.to(device)
            mask_batch = mask_batch.to(device)
            label_batch = label_batch.to(device)
            probs = model(features_batch, mask_batch)
            all_probs.append(probs.cpu())
            all_labels.append(label_batch.cpu())

    endT_test = time.time()

    print(f"Validation inference took {endT_test - startT_test:.2f} seconds.")

    # --------------------------------------------------
    # Collect predictions
    # --------------------------------------------------
    all_probs = torch.cat(all_probs, dim=0).view(-1)
    all_labels = torch.cat(all_labels, dim=0).view(-1)

    endT_test = time.time()

    print(f"\nInference with {val_size} samples took {endT_test - startT_test:.2f} s\n")
    print(f"Predictions shape: {all_probs.shape}, Targets shape: {all_labels.shape}")

    # Plot results
    plotter.plotLabelProbs(all_probs, all_labels)
    plotter.plotTPR_TNR_vs_Threshold(all_probs, all_labels)
    plotter.plotPrecisionRecallF1_vs_Threshold(all_probs, all_labels)

    # Print sample predictions
    for i in range(min(10, all_probs.size(0))):
        print(f"Combo {i}: Prob={all_probs[i]:.3f}, Target={all_labels[i].item()}")


# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == "__main__":
    main()
