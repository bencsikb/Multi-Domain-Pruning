import torch
from torch import nn

class SPN(nn.Module):
    def __init__(self, Nin, Nout):
        super(SPN, self).__init__()

        self.L1 = nn.Linear(Nin, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.L2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.L3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.L4 = nn.Linear(256, Nout)

    def forward(self, x):

        x1 = torch.relu(self.bn1(self.L1(x)))
        x2 = torch.relu(self.bn2(self.L2(x1)))
        x3 = torch.relu(self.bn3(self.L3(x2)))
        x4 = self.L4(x3)

        return x4


class SPNMultihead(nn.Module):
    def __init__(self, Nin):
        super(SPNMultihead, self).__init__()

        # Shared trunk
        self.backend = nn.Sequential(
            nn.Linear(Nin, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # Head for spars
        self.head_spars = nn.Sequential(
            nn.Linear(256, 1)
        )

        # Head for dmap
        self.head_dmap = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        shared_out = self.backend(x)

        out_spars = self.head_spars(shared_out)
        out_dmap = self.head_dmap(shared_out)

        # return shape: (batch, 2)
        return torch.cat([out_spars, out_dmap], dim=1)

class SPNTransformer(nn.Module):
    
    def __init__(self, input_dim=3, model_dim=64, num_heads=4, num_layers=2, dropout=0.1, max_len=100):
        super().__init__()
        self.model_dim = model_dim
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, model_dim))  # learned positional encoding

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(model_dim, 2)  # predicts [spars_next, dmap_next]

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, 3) → [spars, dmap, alpha]
        """
        seq_len = x.size(1)
        x = self.input_proj(x)  # (B, T, model_dim)
        x = x + self.pos_embedding[:, :seq_len, :]  # Add positional encoding
        x = self.transformer(x)  # (B, T, model_dim)
        out = self.output_proj(x)  # (B, T, 2)
        return out