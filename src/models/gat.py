import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, JumpingKnowledge, global_mean_pool
from torch.utils.checkpoint import checkpoint


class GATWithJK(nn.Module):
    """Graph Attention Network with Jumping Knowledge connections."""
    
    def __init__(self, num_ids, in_channels, hidden_channels, out_channels,
                 num_layers=3, heads=4, dropout=0.2, num_fc_layers=3, embedding_dim=8,
                 use_checkpointing=False):
        """Initialize GATWithJK model.

        Args:
            num_ids (int): Number of unique CAN IDs for embedding.
            in_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels.
            out_channels (int): Number of output channels.
            num_layers (int, optional): Number of GAT layers. Defaults to 3.
            heads (int, optional): Number of attention heads. Defaults to 4.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
            num_fc_layers (int, optional): Number of fully connected layers. Defaults to 3.
            embedding_dim (int, optional): Dimension of ID embeddings. Defaults to 8.
            use_checkpointing (bool, optional): Enable gradient checkpointing for memory efficiency. Defaults to False.
        """
        super().__init__()
        self.id_embedding = nn.Embedding(num_ids, embedding_dim)
        self.dropout = dropout
        self.use_checkpointing = use_checkpointing

        # GAT layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = embedding_dim + (in_channels - 1) if i == 0 else hidden_channels * heads
            self.convs.append(GATConv(in_dim, hidden_channels, heads=heads, concat=True))


        self.jk = JumpingKnowledge(
            mode="cat",
            channels=hidden_channels * heads,
            num_layers=num_layers
        )

        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        fc_input_dim = hidden_channels * heads * num_layers
        
        for _ in range(num_fc_layers - 1):
            self.fc_layers.append(nn.Linear(fc_input_dim, fc_input_dim))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(p=dropout))
        self.fc_layers.append(nn.Linear(fc_input_dim, out_channels))

    @classmethod
    def from_config(cls, cfg, num_ids: int, in_ch: int) -> "GATWithJK":
        """Construct from a PipelineConfig."""
        return cls(
            num_ids=num_ids, in_channels=in_ch,
            hidden_channels=cfg.gat.hidden, out_channels=2,
            num_layers=cfg.gat.layers, heads=cfg.gat.heads,
            dropout=cfg.gat.dropout,
            num_fc_layers=cfg.gat.fc_layers,
            embedding_dim=cfg.gat.embedding_dim,
        )

    def forward(self, data, return_intermediate=False):
        """Forward pass through the GATWithJK model.
        
        Args:
            data (torch_geometric.data.Data): Input graph data.
            return_intermediate (bool, optional): Whether to return intermediate representations. Defaults to False.
            
        Returns:
            torch.Tensor: Output predictions or list of intermediate representations if return_intermediate=True.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # x shape: [num_nodes, in_channels], where x[:,0] is CAN ID index
        id_emb = self.id_embedding(x[:, 0].long())  # [num_nodes, embedding_dim]
        other_feats = x[:, 1:]  # [num_nodes, in_channels-1]
        x = torch.cat([id_emb, other_feats], dim=1)


        xs = []
        for conv in self.convs:
            # Apply gradient checkpointing if enabled and in training mode
            if self.use_checkpointing and x.requires_grad:
                # IMPORTANT: Use default args to capture conv and edge_index by value,
                # not by reference. Otherwise, the lambda captures the loop variable
                # which will have changed by the time the backward pass recomputes.
                x = checkpoint(lambda x_in, c=conv, ei=edge_index: c(x_in, ei).relu(), x, use_reentrant=False)
            else:
                x = conv(x, edge_index).relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        if return_intermediate:
            return xs
        x = self.jk(xs)
        x = global_mean_pool(x, batch)
        for layer in self.fc_layers:
            x = layer(x)
        return x

