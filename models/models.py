import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATConv, JumpingKnowledge

class GATWithJK(nn.Module):
    """Graph Attention Network with Jumping Knowledge connections."""
    
    def __init__(self, num_ids, in_channels, hidden_channels, out_channels, 
                 num_layers=3, heads=4, dropout=0.2, num_fc_layers=3, embedding_dim=8):
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
        """
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
class GraphAutoencoderNeighborhood(nn.Module):
    """
    Graph Autoencoder that reconstructs node features and edge list.

    Components:
    - Node Encoder: Uses GATConv layers (with batch norm and dropout) to encode node features and CAN ID embeddings into latent node representations.
    - Latent Space: Outputs mean and logvar for each node, enabling sampling (VGAE style) and KL regularization.
    - Node Decoder: Uses GATConv layers to reconstruct node features from latent embeddings.
    - CAN ID Classifier: Linear layer to predict CAN ID for each node from latent features.
    - Edge Decoder: MLP that predicts edge existence probability between node pairs, using both node embeddings and (optionally) edge attributes.

    Forward Flow:
    1. **Encoding:** Node features (including CAN ID embedding) are passed through several GATConv layers to produce hidden node representations. These are then mapped to mean and logvar vectors, and sampled to produce latent node embeddings (`z`). KL divergence is computed for regularization.
    2. **Node Decoding:** Latent node embeddings are passed through GATConv layers to reconstruct the original node features (except CAN ID, which is handled separately).
    3. **CAN ID Classification:** Latent node embeddings are also used to predict CAN ID for each node.
    4. **Edge Decoding:** For each edge (or candidate edge), the latent embeddings of the source and target nodes are combined (concat, hadamard, L1 distance), optionally concatenated with edge attributes, and passed through an MLP to predict the probability that the edge exists.
    5. **Losses:** Node reconstruction loss, CAN ID classification loss, edge reconstruction loss, and KL loss are combined during training.

    The model is trained to reconstruct both node features and the graph structure (edges), and can use rich edge attributes to improve edge prediction.
    """
    def __init__(self, num_ids, in_channels, hidden_dim=32, latent_dim=32,
                  heads=4, embedding_dim=8, dropout=0.35):
        """Initialize GraphAutoencoder.
        
        Args:
            num_ids (int): Number of unique CAN IDs for embedding.
            in_channels (int): Number of input channels.
            hidden_dim (int, optional): Hidden dimension size. Defaults to 32.
            latent_dim (int, optional): Latent dimension size. Defaults to 32.
            heads (int, optional): Number of attention heads. Defaults to 4.
            embedding_dim (int, optional): Dimension of ID embeddings. Defaults to 8.
            dropout (float, optional): Dropout rate. Defaults to 0.35.
        """
        super().__init__()
        self.id_embedding = nn.Embedding(num_ids, embedding_dim)
        self.latent_dim = latent_dim
        gat_in_dim = embedding_dim + (in_channels - 1)
        # ADD THIS LINE - Store num_ids as instance attribute
        self.num_ids = num_ids

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

        # Add neighborhood decoder
        self.neighborhood_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_ids)  # Output: [num_nodes, num_can_ids]
        )

        # Decoder for node features: 3 GAT layers
        self.dec1 = GATConv(hidden_dim, hidden_dim, heads=heads)
        self.dbn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.dec2 = GATConv(hidden_dim * heads, hidden_dim, heads=1)
        self.dbn2 = nn.BatchNorm1d(hidden_dim)
        self.dec3 = GATConv(hidden_dim, in_channels - 1, heads=1)  # Exclude CAN ID from continuous output

        # CAN ID classifier head
        self.canid_classifier = nn.Linear(hidden_dim, num_ids)
        self.gat_in_dim = gat_in_dim


    def encode(self, x, edge_index):
        """Encode input graph into latent representation.
        
        Args:
            x (torch.Tensor): Node features with shape [num_nodes, in_channels].
            edge_index (torch.Tensor): Edge indices with shape [2, num_edges].
            
        Returns:
            tuple: (latent_embeddings, kl_loss) where latent_embeddings has shape [num_nodes, latent_dim].
        """
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
        """Decode latent representation back to node features.
        
        Args:
            z (torch.Tensor): Latent node embeddings with shape [num_nodes, latent_dim].
            edge_index (torch.Tensor): Edge indices with shape [2, num_edges].
            
        Returns:
            tuple: (continuous_output, canid_logits) for node feature reconstruction and CAN ID classification.
        """
        x1 = self.dropout(F.relu(self.dbn1(self.dec1(z, edge_index))))
        x2 = self.dropout(F.relu(self.dbn2(self.dec2(x1, edge_index))))
        cont_out = torch.sigmoid(self.dec3(x2, edge_index))  # shape: [num_nodes, in_channels-1]
        canid_logits = self.canid_classifier(x2)  # shape: [num_nodes, num_ids]
        return cont_out, canid_logits
    
    def decode_neighborhood(self, z):
        """Decode latent representation to neighborhood predictions.
        
        Args:
            z (torch.Tensor): Latent node embeddings with shape [num_nodes, latent_dim].
            
        Returns:
            torch.Tensor: Neighborhood logits with shape [num_nodes, num_ids].
        """
        neighbor_logits = self.neighborhood_decoder(z)
        return neighbor_logits
    
    def create_neighborhood_targets(self, x, edge_index, batch):
        """Create neighborhood target matrix for training.
        
        Args:
            x (torch.Tensor): Node features with CAN IDs in first column.
            edge_index (torch.Tensor): Edge indices.
            batch (torch.Tensor): Batch assignment vector.
            
        Returns:
            torch.Tensor: Binary target matrix [num_nodes, num_ids].
        """
        num_nodes = x.size(0)
        device = x.device
        
        # Initialize target matrix
        neighbor_targets = torch.zeros(num_nodes, self.num_ids, device=device)
        
        # Fill in actual neighbors based on edges
        for i in range(edge_index.size(1)):
            src_node, dst_node = edge_index[0, i], edge_index[1, i]
            dst_can_id = x[dst_node, 0].long()  # CAN ID of destination node
            
            # Set target: src_node should predict dst_can_id as neighbor
            if dst_can_id < self.num_ids:  # Ensure valid CAN ID
                neighbor_targets[src_node, dst_can_id] = 1.0
        
        return neighbor_targets

    def forward(self, x, edge_index, batch):
        """Forward pass through the GraphAutoencoderNeighborhood.
        
        Args:
            x (torch.Tensor): Node features with shape [num_nodes, in_channels].
            edge_index (torch.Tensor): Edge indices with shape [2, num_edges].
            batch (torch.Tensor): Batch assignment vector.
            
        Returns:
            tuple: (continuous_output, canid_logits, neighbor_logits, latent_embeddings, kl_loss).
        """
        z, kl_loss = self.encode(x, edge_index)
        cont_out, canid_logits = self.decode_node(z, edge_index)
        neighbor_logits = self.decode_neighborhood(z)
        return cont_out, canid_logits, neighbor_logits, z, kl_loss



if __name__ == '__main__':
    # Knowledge Distillation Scenario
    # teacher_model = GATWithJK(in_channels=10, hidden_channels=32, out_channels=1, num_layers=5, heads=8)
    # student_model = GATWithJK(in_channels=10, hidden_channels=32, out_channels=1, num_layers=2, heads=4)
    autoencoder = GraphAutoencoderNeighborhood(num_ids=2000, in_channels=11, embedding_dim=8)
    # net = GATWithJK(10, 8, 1)

    def model_characteristics(model):
        """Print model characteristics including parameter count and size.
        
        Args:
            model (torch.nn.Module): PyTorch model to analyze.
        """
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