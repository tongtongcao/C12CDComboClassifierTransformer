
import numpy as np
from typing import Final
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


# --------------------------------------------------
# Configuration
# --------------------------------------------------

LAYER_IDS = [1, 2, 3, 4, 7, 9, 5, 6, 8]

# Token indices in the fixed sequence.
SVT_TOKENS = [0, 1, 2]
BMTC_TOKENS = [3, 4, 5]
BMTZ_TOKENS = [6, 7, 8]

# --------------------------------------------------
# Dataset
# --------------------------------------------------

class comboDataset(Dataset):
    """
    Dataset for cross-combination classification.

    Each sample contains:

        features:   [9, 3]
        valid_mask: [9], True = valid detector layer
        label:      [1]

    Token representation:

        SVT:  (x, y, z)
        BMTC: (r, z, 0)       # Third coordinate is ignored
        BMTZ: (r, sin(phi), cos(phi))

    Feature normalization is performed independently
    for each token and feature.
    """

    def __init__(
        self,
        features,
        valid_mask,
        labels,
        normalize=True,
        feature_stats=None,
    ):

        assert len(features) == len(valid_mask) == len(labels)

        self.features = np.asarray(features, dtype=np.float32)
        self.valid_mask = np.asarray(valid_mask, dtype=bool)
        self.labels = np.asarray(labels, dtype=np.float32).reshape(-1)

        if self.features.ndim != 3 or self.features.shape[1:] != (9, 3):
            raise ValueError(f"Expected features shape [N, 9, 3], got {self.features.shape}")

        if self.valid_mask.shape != (len(self.features), 9):
            raise ValueError("Invalid mask shape.")

        if self.labels.shape != (len(self.features),):
            raise ValueError(f"Invalid label shape: {self.labels.shape}")

        if not np.all(self.valid_mask.any(axis=1)):
            raise ValueError("Every track must have at least one valid token.")

        if not np.all(np.isfinite(self.features)):
            raise ValueError("Non-finite input features detected.")

        if not np.all(np.isin(self.labels, [0.0, 1.0])):
            raise ValueError("Labels must be binary: 0 or 1.")

        self.normalize = normalize

        # Feature statistics: [9, 3].
        self.feature_stats = feature_stats or {
            "mean": np.zeros((9, 3), dtype=np.float32),
            "std": np.ones((9, 3), dtype=np.float32)
        }

        self.feature_mean = np.asarray(self.feature_stats["mean"], dtype=np.float32)

        self.feature_std = np.asarray(self.feature_stats["std"], dtype=np.float32)

        if self.feature_mean.shape != (9, 3):
            raise ValueError("Feature mean must have shape [9, 3].")

        if self.feature_std.shape != (9, 3):
            raise ValueError("Feature std must have shape [9, 3].")

        if not np.all(np.isfinite(self.feature_mean)):
            raise ValueError("Invalid feature means.")

        if not np.all(np.isfinite(self.feature_std)):
            raise ValueError("Invalid feature standard deviations.")

        if np.any(self.feature_std <= 0):
            raise ValueError("Feature standard deviations must be positive.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        features = self.features[idx].copy()
        valid_mask = self.valid_mask[idx].copy()
        label = self.labels[idx].copy()

        if self.normalize:

            features = (features - self.feature_mean) / self.feature_std

            # Invalid tokens remain zero.
            features[~valid_mask] = 0.0

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(valid_mask, dtype=torch.bool),
            torch.tensor(label, dtype=torch.float32)
        )

# --------------------------------------------------
# Collate function
# --------------------------------------------------

def collate_fn(batch):

    features, valid_masks, labels = zip(*batch)

    return (
        torch.stack(features),
        torch.stack(valid_masks),
        torch.stack(labels)
    )


# --------------------------------------------------
# Transformer model
# --------------------------------------------------

class comboClassifierTransformer(pl.LightningModule):
    """
    Transformer-based cross-combination classification.

    Input:
        x:          [B, 9, 3]
        valid_mask: [B, 9], True = valid

    Output:
        Probability

    Detector-specific input embeddings:

        SVT:  3 -> hidden_dim
        BMTC: 2 -> hidden_dim
        BMTZ: 3 -> hidden_dim
    """
    NUM_TOKENS: Final[int] = 9

    def __init__(self, hidden_dim=32, nhead=4, num_layers=2, lr=1e-3, pos_weight=1):

        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.pos_weight = pos_weight

        # --------------------------------------------------
        # Detector-specific feature embeddings
        # --------------------------------------------------

        self.svt_embedding = nn.Linear(3, hidden_dim)
        self.bmtc_embedding = nn.Linear(2, hidden_dim)
        self.bmtz_embedding = nn.Linear(3, hidden_dim)

        # --------------------------------------------------
        # Learnable layer embeddings
        # --------------------------------------------------

        self.layer_embedding = nn.Embedding(self.NUM_TOKENS, hidden_dim)

        # --------------------------------------------------
        # Transformer encoder
        # --------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4, batch_first=True)

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --------------------------------------------------
        # Output probability for binary classification
        # --------------------------------------------------

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.loss_fn = nn.BCELoss(reduction='none')

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    def forward(self, x, valid_mask):
        """
        x:          [B, 9, 3]
        valid_mask: [B, 9], True = valid
        """

        # Detector-specific feature embedding.
        #
        # SVT:  [B, 3, 3] -> [B, 3, H]
        # BMTC: [B, 3, 2] -> [B, 3, H]
        # BMTZ: [B, 3, 3] -> [B, 3, H]

        svt = self.svt_embedding(x[:, 0:3, :])
        bmtc = self.bmtc_embedding(x[:, 3:6, 0:2])
        bmtz = self.bmtz_embedding(x[:, 6:9, :])

        # Reassemble tokens in the original fixed order.
        x_emb = torch.cat([svt, bmtc, bmtz], dim=1)

        # Add layer identity.
        layer_ids = torch.arange(self.NUM_TOKENS, device=x.device)
        layer_emb = self.layer_embedding(layer_ids)

        x_emb = x_emb + layer_emb.unsqueeze(0)

        # True in src_key_padding_mask means ignore.
        padding_mask = ~valid_mask

        # Transformer encoder.
        x_trans = self.transformer(x_emb, src_key_padding_mask=padding_mask)

        # Masked mean pooling.
        valid_weights = valid_mask.unsqueeze(-1).to(x_trans.dtype)
        x_trans = x_trans * valid_weights
        num_valid = valid_weights.sum(dim=1).clamp(min=1.0)
        x_pooled = x_trans.sum(dim=1) / num_valid

        # classification
        return self.fc(x_pooled)

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------
    def compute_loss(self, y_hat, y):
        y_hat = y_hat.reshape(-1)
        y = y.float().reshape(-1)
        loss = self.loss_fn(y_hat, y)
        weights = torch.where(y == 1, torch.full_like(y, self.pos_weight), torch.ones_like(y))

        return (loss * weights).mean()

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    def training_step(self, batch, batch_idx):
        """
        Training step for a single batch.

        Parameters
        ----------
        batch : tuple
            Tuple of (inputs, targets).
        batch_idx : int
            Index of the batch.

        Returns
        -------
        torch.Tensor
            Training loss for the batch.
        """
        x, valid_mask, labels = batch
        y_hat = self(x, valid_mask)
        loss = self.compute_loss(y_hat, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    def validation_step(self, batch, batch_idx):
        """
        Validation step for a single batch.

        Parameters
        ----------
        batch : tuple
            Tuple of (inputs, targets).
        batch_idx : int
            Index of the batch.
        """
        x, valid_mask, labels = batch
        y_hat = self(x, valid_mask)
        loss = self.compute_loss(y_hat, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# --------------------------------------------------
# TorchScript inference wrapper
# --------------------------------------------------

class comboClassifierTransformerWrapper(nn.Module):
    """
    Inference interface:

        features:   [B, 9, 3]
        valid_mask: [B, 9]

    Returns:
        probability: [B, 1]
    """

    def __init__(self, core_model):
        super().__init__()
        self.core = core_model

    def forward(self, x, valid_mask):
        return self.core(x, valid_mask)


# --------------------------------------------------
# Loss tracking callback
# --------------------------------------------------

class LossTracker(Callback):

    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        if "train_loss" in trainer.callback_metrics:
            self.train_losses.append(trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        if "val_loss" in trainer.callback_metrics:
            self.val_losses.append(trainer.callback_metrics["val_loss"].item())
