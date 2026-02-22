import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class Attention(nn.Module):
    """
    Attention mechanism that computes a weighted sum of the LSTM outputs.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: [batch_size, seq_len, hidden_dim * 2]
        Returns:
            context: [batch_size, hidden_dim * 2]
            attention_weights: [batch_size, seq_len]
        """
        # Calculate attention scores
        attention_scores = self.attention(lstm_output).squeeze(2)  # [batch_size, seq_len]
        attention_weights = self.softmax(attention_scores)  # [batch_size, seq_len]
        
        # Calculate context vector
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)  # [batch_size, hidden_dim * 2]
        
        return context, attention_weights

class SuspiciousActivityLSTM(nn.Module):
    """
    Bidirectional LSTM with Attention for suspicious activity detection.
    Takes pose features as input and outputs a binary classification.
    """
    def __init__(
        self,
        input_size: int = 204,  # 102 features per person * 2 people
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = Attention(hidden_size)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size * (2 if bidirectional else 1), 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                elif 'attention' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'fc' in name:
                    nn.init.kaiming_normal_(param.data, mode='fan_in', nonlinearity='leaky_relu')
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size]
            return_attention: If True, also return attention weights
            
        Returns:
            output: Output logits of shape [batch_size, 1]
            attention_weights: (Optional) Attention weights of shape [batch_size, seq_len]
        """
        batch_size = x.size(0)
        
        # Initialize hidden state and cell state
        h0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            batch_size,
            self.hidden_size
        ).to(x.device)
        
        c0 = torch.zeros_like(h0)
        
        # LSTM forward
        lstm_out, _ = self.lstm(x, (h0, c0))  # [batch_size, seq_len, hidden_size * 2]
        
        # Apply attention if enabled
        if self.use_attention:
            context, attention_weights = self.attention(lstm_out)
            x = context
        else:
            # Use the last hidden state if attention is disabled
            if self.bidirectional:
                # Concatenate the forward and backward final hidden states
                forward_hidden = lstm_out[:, -1, :self.hidden_size]
                backward_hidden = lstm_out[:, 0, self.hidden_size:]
                x = torch.cat([forward_hidden, backward_hidden], dim=1)
            else:
                x = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)  # [batch_size, 1]
        
        if return_attention and self.use_attention:
            return logits, attention_weights
        return logits
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for the input sequence.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size]
            
        Returns:
            attention_weights: Attention weights of shape [batch_size, seq_len]
        """
        if not self.use_attention:
            raise ValueError("Attention is not enabled for this model")
        
        with torch.no_grad():
            _, attention_weights = self.forward(x, return_attention=True)
        
        return attention_weights

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test the model with a sample input
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a sample input
    batch_size = 4
    seq_len = 30  # Number of frames
    input_size = 204  # 102 features per person * 2 people
    
    x = torch.randn(batch_size, seq_len, input_size).to(device)
    
    # Initialize model
    model = SuspiciousActivityLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
        use_attention=True
    ).to(device)
    
    # Test forward pass
    output, attention_weights = model(x, return_attention=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Number of trainable parameters: {count_parameters(model):,}")
