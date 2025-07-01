import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATConv, JumpingKnowledge
# class GATWithJK(torch.nn.Module):
#     def __init__(self, num_ids, in_channels, hidden_channels, out_channels, 
#                  num_layers=3, heads=4, dropout=0.2, num_fc_layers=3, embedding_dim=8):
#         super().__init__()
#         self.id_embedding = nn.Embedding(num_ids, embedding_dim)
#         self.convs = torch.nn.ModuleList()
#         self.dropout = dropout
        
#         # GAT layers
#         for i in range(num_layers):
#             in_dim = in_channels if i == 0 else hidden_channels * heads
#             self.convs.append(
#                 GATConv(in_dim, hidden_channels, heads=heads, concat=True)
#             )
        
#         # JK aggregation (LSTM mode)
#         self.jk = JumpingKnowledge(
#             # will try cat for speed and lower memory usage
#             mode="cat", # "cat" | "max" | "mean" | "lstm"
#             channels=hidden_channels * heads,
#             num_layers=num_layers
#         )
        
#         # Fully connected layers
#         self.fc_layers = torch.nn.ModuleList()
#         fc_input_dim = hidden_channels * heads
#         for _ in range(num_fc_layers - 1):
#             self.fc_layers.append(torch.nn.Linear(fc_input_dim, fc_input_dim))
#             self.fc_layers.append(torch.nn.ReLU())
#             self.fc_layers.append(torch.nn.Dropout(p=dropout))
#         self.fc_layers.append(torch.nn.Linear(fc_input_dim, out_channels))
        
        

#     def forward(self, data, return_intermediate=False):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         xs = []
#         for conv in self.convs:
#             x = conv(x, edge_index).relu()
#             x = F.dropout(x, p=self.dropout, training=self.training)  # Add dropout
#             xs.append(x)
        
#         if return_intermediate:
#             return xs
        
#         # Aggregate layer outputs
#         x = self.jk(xs)
#         x = global_mean_pool(x, batch)  # Readout layer
#         # Pass through fully connected layers + final output layer
#         for layer in self.fc_layers:
#             x = layer(x)
        
#         return x

class GATWithJK(nn.Module):
    def __init__(self, num_ids, in_channels, hidden_channels, out_channels, 
                 num_layers=3, heads=4, dropout=0.2, num_fc_layers=3, embedding_dim=8):
        super().__init__()
        self.id_embedding = nn.Embedding(num_ids, embedding_dim)
        self.dropout = dropout

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
        if self.jk.mode == "cat":
            fc_input_dim = hidden_channels * heads * num_layers
        else:
            fc_input_dim = hidden_channels * heads
        
        for _ in range(num_fc_layers - 1):
            # rotuer top2 experts
            self.fc_layers.append(nn.Linear(fc_input_dim, fc_input_dim))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(p=dropout))
        self.fc_layers.append(nn.Linear(fc_input_dim, out_channels))

    def forward(self, data, return_intermediate=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # x shape: [num_nodes, in_channels], where x[:,0] is CAN ID index
        id_emb = self.id_embedding(x[:, 0].long())  # [num_nodes, embedding_dim]
        other_feats = x[:, 1:]  # [num_nodes, in_channels-1]
        x = torch.cat([id_emb, other_feats], dim=1)


        xs = []
        for conv in self.convs:
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
class GATMoEJK(nn.Module):
    def __init__(self, num_ids, in_channels, hidden_channels, out_channels, 
                 num_layers=3, heads=4, dropout=0.2, num_fc_layers=3, embedding_dim=8):
        super().__init__()
        self.id_embedding = nn.Embedding(num_ids, embedding_dim)
        self.dropout = dropout

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
        if self.jk.mode == "cat":
            fc_input_dim = hidden_channels * heads * num_layers
        else:
            fc_input_dim = hidden_channels * heads
        
        # TODO: Add MoE layers
        for _ in range(num_fc_layers - 1):
            # rotuer top2 experts
            self.fc_layers.append(nn.Linear(fc_input_dim, fc_input_dim))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(p=dropout))
        self.fc_layers.append(nn.Linear(fc_input_dim, out_channels))

    def forward(self, data, return_intermediate=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # x shape: [num_nodes, in_channels], where x[:,0] is CAN ID index
        id_emb = self.id_embedding(x[:, 0].long())  # [num_nodes, embedding_dim]
        other_feats = x[:, 1:]  # [num_nodes, in_channels-1]
        x = torch.cat([id_emb, other_feats], dim=1)


        xs = []
        for conv in self.convs:
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

class GraphAutoencoder(nn.Module):
    """Graph Autoencoder: reconstructs node features and edge list."""
    def __init__(self, num_ids, in_channels, hidden_dim=32, latent_dim=32,
                  heads=4, embedding_dim=8, dropout=0.35):
        super().__init__()
        self.id_embedding = nn.Embedding(num_ids, embedding_dim)
        self.latent_dim = latent_dim
        gat_in_dim = embedding_dim + (in_channels - 1)

        # Encoder: 3 GAT layers with batch norm and residuals
        self.enc1 = GATConv(gat_in_dim, hidden_dim, heads=heads)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.enc2 = GATConv(hidden_dim * heads, hidden_dim, heads=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.enc3 = GATConv(hidden_dim, hidden_dim, heads=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

        # Add mean and logvar heads for z, sample z ~ N(mu, sigma)
        self.z_mean = nn.Linear(hidden_dim, latent_dim)
        self.z_logvar = nn.Linear(hidden_dim, latent_dim)

        # Create edge decoder dynamically - will be initialized on first forward pass
        self.edge_decoder = None
        self.dropout_rate = dropout

        # Decoder for node features: 3 GAT layers
        self.dec1 = GATConv(hidden_dim, hidden_dim, heads=heads)
        self.dbn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.dec2 = GATConv(hidden_dim * heads, hidden_dim, heads=1)
        self.dbn2 = nn.BatchNorm1d(hidden_dim)
        self.dec3 = GATConv(hidden_dim, in_channels - 1, heads=1)  # Exclude CAN ID from continuous output

        # CAN ID classifier head
        self.canid_classifier = nn.Linear(hidden_dim, num_ids)
        self.gat_in_dim = gat_in_dim

    def _build_edge_decoder(self, latent_dim, num_layers=3, dropout=0.35):
        """Build a more sophisticated edge decoder with configurable depth"""
        layers = nn.ModuleList()
        
        # Input dimension: concatenated node embeddings + interaction features
        input_dim = 3 * latent_dim  # concat + hadamard + l1_distance
        hidden_dims = [128, 64, 32][:num_layers-1] + [1]
        
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            if i < len(hidden_dims) - 1:  # Hidden layers
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                # Add residual connection if dimensions match
                if prev_dim == hidden_dim:
                    layers.append(nn.Identity())  # Placeholder for residual
            else:  # Output layer
                layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        return nn.Sequential(*layers)

    def _compute_edge_features(self, z, edge_index):
        """Compute rich edge features from node embeddings"""
        z_src = z[edge_index[0]]  # [num_edges, latent_dim]
        z_dst = z[edge_index[1]]  # [num_edges, latent_dim]
        
        # Multiple ways to combine node embeddings
        concat_feat = torch.cat([z_src, z_dst], dim=1)  # [num_edges, 2*latent_dim]
        hadamard_feat = z_src * z_dst  # Element-wise product [num_edges, latent_dim]
        l1_distance = torch.abs(z_src - z_dst)  # L1 distance [num_edges, latent_dim]
        
        # Combine all features
        edge_feat = torch.cat([concat_feat, hadamard_feat, l1_distance], dim=1)
        return edge_feat

    def _forward_edge_decoder_with_residual(self, edge_feat):
        """Forward pass through edge decoder with residual connections"""
        x = edge_feat
        residual = None
        
        for i, layer in enumerate(self.edge_decoder):
            if isinstance(layer, nn.Linear):
                if residual is not None and x.size(-1) == residual.size(-1):
                    x = x + residual  # Add residual connection
                x = layer(x)
                if i < len(self.edge_decoder) - 1:  # Not the last layer
                    residual = x.clone()
            elif isinstance(layer, nn.Identity):
                continue  # Skip identity layers
            else:
                x = layer(x)
        
        return x

    def encode(self, x, edge_index):
        id_emb = self.id_embedding(x[:, 0].long())
        other_feats = x[:, 1:]
        x = torch.cat([id_emb, other_feats], dim=1)
        x1 = self.dropout(F.relu(self.bn1(self.enc1(x, edge_index))))
        x2 = self.dropout(F.relu(self.bn2(self.enc2(x1, edge_index))))
        # Residual connection
        x3 = self.dropout(F.relu(self.bn3(self.enc3(x2, edge_index))))
        h = x3 + x2  # Residual

        # VGAE Latent Regularization
        mu = self.z_mean(h)
        logvar = self.z_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # Reparameterization trick
        # KL divergence loss (mean over all nodes)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return z, kl_loss

    def decode_node(self, z, edge_index):
        x1 = self.dropout(F.relu(self.dbn1(self.dec1(z, edge_index))))
        x2 = self.dropout(F.relu(self.dbn2(self.dec2(x1, edge_index))))
        cont_out = torch.sigmoid(self.dec3(x2, edge_index))  # shape: [num_nodes, in_channels-1]
        canid_logits = self.canid_classifier(x2)  # shape: [num_nodes, num_ids]
        return cont_out, canid_logits

    def _create_edge_decoder(self, edge_feat_dim):
        """Create edge decoder with correct input dimensions"""
        return nn.Sequential(
            nn.Linear(edge_feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(32, 1)
        ).to(next(self.parameters()).device)

    def decode_edge(self, z, edge_index):
        """Enhanced edge decoder with richer features"""
        z_src = z[edge_index[0]]  # [num_edges, latent_dim]
        z_dst = z[edge_index[1]]  # [num_edges, latent_dim]
        
        # Multiple ways to combine node embeddings
        concat_feat = torch.cat([z_src, z_dst], dim=1)  # [num_edges, 2*latent_dim]
        hadamard_feat = z_src * z_dst  # Element-wise product [num_edges, latent_dim]
        l1_distance = torch.abs(z_src - z_dst)  # L1 distance [num_edges, latent_dim]
        
        # Combine all features: [num_edges, ?*latent_dim]
        edge_feat = torch.cat([concat_feat, hadamard_feat, l1_distance], dim=1)
        
        # Create edge decoder on first use with correct dimensions
        if self.edge_decoder is None:
            print(f"Creating edge decoder with input dimension: {edge_feat.size(1)}")
            self.edge_decoder = self._create_edge_decoder(edge_feat.size(1))
        
        # Forward through decoder
        edge_logits = self.edge_decoder(edge_feat).squeeze(-1)
        edge_probs = torch.sigmoid(edge_logits)
        return edge_probs

    def forward(self, x, edge_index, batch):
        z, kl_loss = self.encode(x, edge_index)
        cont_out, canid_logits = self.decode_node(z, edge_index)
        return cont_out, canid_logits, z, kl_loss


'''
Variational Graph Autoencoder (VGAE) Latent Regularization
# Add mean and logvar heads for z, sample z ~ N(mu, sigma)
self.z_mean = nn.Linear(hidden_dim, latent_dim)
self.z_logvar = nn.Linear(hidden_dim, latent_dim)
# In encode():
mu = self.z_mean(x)
logvar = self.z_logvar(x)
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
z = mu + eps * std
# Add KL loss to your total loss
kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
'''
if __name__ == '__main__':
    # Knowledge Distillation Scenario
    # teacher_model = GATWithJK(in_channels=10, hidden_channels=32, out_channels=1, num_layers=5, heads=8)
    # student_model = GATWithJK(in_channels=10, hidden_channels=32, out_channels=1, num_layers=2, heads=4)
    autoencoder = GraphAutoencoder(num_ids=2000, in_channels=11, embedding_dim=8)
    # net = GATWithJK(10, 8, 1)

    def model_characteristics(model):
        num_params = sum(p.numel() for p in model.parameters())
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        print(f'Number of Parameters: {num_params:.3f}')
        print(f'Model size: {size_all_mb:.3f} MB')

    # model_characteristics(teacher_model)
    # print(teacher_model)
    # model_characteristics(student_model)
    # print(student_model)
    model_characteristics(autoencoder)
    print(autoencoder)