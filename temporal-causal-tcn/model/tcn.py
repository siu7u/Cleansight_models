from pytorch_tcn import TCN
import torch.nn as nn

class TCNClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=(64, 64, 64)):
        super().__init__()
        self.tcn = TCN(
            num_inputs=input_dim,
            num_channels=list(hidden_dims),
            kernel_size=3,
            dropout=0.2,
            causal=True,
            use_norm='weight_norm',
            activation='relu',
        )
        self.head = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x):
        # x: (B, T, F)  batch_first
        x = x.transpose(1, 2)     # (B, F, T) for Conv1d
        feat = self.tcn(x)        # (B, C, T)
        feat = feat.transpose(1, 2)  # (B, T, C)
        return self.head(feat)    # 逐帧 logits (B, T, num_classes)