"""
Bi-LSTM + Multi-Head Attention violence detection architecture.

Key design principle: Separate individual motion patterns from interaction context.
This prevents false positives on high-motion sports while detecting actual violence.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import PERSON_DIM, INTERACTION_DIM, FEATURE_DIM, SEQ_LEN


class ViolenceDetectorV3(nn.Module):
    """
    Bi-LSTM + Multi-Head Attention violence detector.
    
    Architecture:
        Input (126D) → Person encoder → Interaction encoder → Merge
        → Bi-LSTM (bidirectional) → LayerNorm → Multi-Head Attention
        → Max pool → Classifier
    
    Key features:
      - Separate encoding for person and interaction features
      - Bidirectional LSTM for temporal context
      - Multi-head attention to focus on key frames
      - Classifier head for binary decision
    
    Model size: ~91K parameters, 0.36 MB FP32, ~91 KB INT8
    Input: (batch, seq_len=60, features=126)
    Output: (batch, 1) logit + attention weights
    """
    
    def __init__(self, input_dim=FEATURE_DIM, person_dim=PERSON_DIM, hidden_dim=64,
                 num_heads=4, num_layers=2, dropout=0.4):
        """
        Args:
            input_dim: Total input feature dimension (126)
            person_dim: Per-person feature dimension (60)
            hidden_dim: LSTM hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        inter_dim = input_dim - 2 * person_dim  # Interaction dimension = 6
        
        # ===== Person Encoders =====
        # Separate encoding for each person's features
        self.person_enc = nn.Sequential(
            nn.Linear(person_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # ===== Interaction Encoder =====
        # Capture distance, approach velocity, IoU, etc.
        self.inter_enc = nn.Sequential(
            nn.Linear(inter_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )
        
        # ===== Merge Layer =====
        # Combine person and interaction representations
        self.merge = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 32, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU()
        )
        
        # ===== Bi-LSTM =====
        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.lstm_norm = nn.LayerNorm(hidden_dim * 2)
        
        # ===== Multi-Head Attention =====
        # Focus on key temporal frames
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0
        )
        self.attn_norm = nn.LayerNorm(hidden_dim * 2)
        
        # ===== Classifier =====
        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
        
        self.person_dim = person_dim
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, 126)
                First 60D: Person A features
                Next 60D: Person B features
                Last 6D: Interaction features
        
        Returns:
            Tuple of (logit, attention_weights)
                logit: (batch, 1) - raw output before sigmoid
                attention_weights: (batch, seq_len, seq_len) - attention matrix
        """
        # Split features
        pA = x[:, :, :self.person_dim]              # (batch, seq, 60)
        pB = x[:, :, self.person_dim:self.person_dim * 2]  # (batch, seq, 60)
        inter = x[:, :, self.person_dim * 2:]       # (batch, seq, 6)
        
        # Encode persons (apply to each timestep)
        hA = self.person_enc(pA)  # (batch, seq, hidden_dim)
        hB = self.person_enc(pB)
        
        # Encode interactions
        hi = self.inter_enc(inter)  # (batch, seq, 32)
        
        # Merge encodings
        h = self.merge(torch.cat([hA, hB, hi], dim=-1))  # (batch, seq, hidden_dim*2)
        
        # Bi-LSTM
        h, _ = self.lstm(h)  # (batch, seq, hidden_dim*2)
        h = self.lstm_norm(h)
        
        # Multi-head attention
        attn_out, attn_weights = self.attn(h, h, h)  # (batch, seq, hidden_dim*2)
        h = self.attn_norm(attn_out + h)  # Residual connection
        
        # Max pool over sequence
        h_pooled = torch.max(h, dim=1)[0]  # (batch, hidden_dim*2)
        
        # Classify
        logit = self.classifier(h_pooled)  # (batch, 1)
        
        return logit, attn_weights


class FocalLoss(nn.Module):
    """
    Focal Loss with label smoothing.
    
    Helps with:
      - Class imbalance (α=0.75 weights fight class higher)
      - Hard examples (γ=2.0 focuses on boundary cases)
      - Overfitting on training domains (smoothing prevents overconfidence)
    """
    
    def __init__(self, alpha=0.75, gamma=2.0, smoothing=0.05):
        """
        Args:
            alpha: Balance factor for positive class (fight)
            gamma: Focusing parameter (0=CE, 2=focus on hard)
            smoothing: Label smoothing factor (prevents overconfidence)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
    
    def forward(self, logits, targets):
        """
        Compute focal loss.
        
        Args:
            logits: Raw model outputs (batch,)
            targets: Binary targets 0 or 1 (batch,)
        
        Returns:
            Scalar loss
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        
        # Label smoothing
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        # BCE loss
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Focal term
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        
        return (focal_weight * bce).mean()
