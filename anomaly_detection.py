import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.utils import unbatch

class Autoencoder(nn.Module):
    """Processes time-series graphs through GAT layers"""
    def __init__(self, time_dim, hidden_dim=32, heads=4):
        super().__init__()
        # Encoder
        self.enc1 = GATConv(time_dim, hidden_dim, heads=heads)
        self.enc2 = GATConv(hidden_dim*heads, hidden_dim, heads=1)
        
        # Decoder 
        self.dec1 = GATConv(hidden_dim, hidden_dim, heads=heads)
        self.dec2 = GATConv(hidden_dim*heads, time_dim, heads=1)

    def forward(self, x, edge_index):
        # Encoding
        x = self.enc1(x, edge_index).relu()
        x = self.enc2(x, edge_index).relu()
        
        # Decoding
        x = self.dec1(x, edge_index).relu()
        return self.dec2(x, edge_index).sigmoid()

class Classifier(nn.Module):
    """Classification with temporal attention"""
    def __init__(self, time_dim, hidden_dim=32, heads=4):
        super().__init__()
        self.gat1 = GATConv(time_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim*heads, hidden_dim, heads=1)
        self.class_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        # Graph processing
        x = self.gat1(x, edge_index).relu()
        x = self.gat2(x, edge_index).relu()
        
        # Temporal pooling and classification
        return self.class_head(global_mean_pool(x, batch)).sigmoid()

class GATPipeline:
    def __init__(self, time_dim, device='cpu'):
        self.device = device
        self.autoencoder = Autoencoder(time_dim).to(device)
        self.classifier = Classifier(time_dim).to(device)
        self.threshold = None

    def train_stage1(self, train_loader, epochs=50):
        """Train on normal time-series graphs"""
        self.autoencoder.train()
        opt = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-3)
        
        for _ in range(epochs):
            for batch in train_loader:
                batch = batch.to(self.device)
                recon = self.autoencoder(batch.x, batch.edge_index)
                loss = nn.MSELoss()(recon, batch.x)
                
                opt.zero_grad()
                loss.backward()
                opt.step()

        # Set reconstruction threshold (95th percentile)
        errors = []
        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(self.device)
                recon = self.autoencoder(batch.x, batch.edge_index)
                errors.append((recon - batch.x).pow(2).mean(dim=1))
        
        self.threshold = torch.cat(errors).quantile(0.90).item()

    def train_stage2(self, full_loader, epochs=50):
        """Train classifier on filtered graphs"""
        filtered = []
        
        # Filter using autoencoder
        with torch.no_grad():
            for batch in full_loader:
                batch = batch.to(self.device)
                recon = self.autoencoder(batch.x, batch.edge_index)
                error = (recon - batch.x).pow(2).mean(dim=1)
                
                # Process each graph individually
                for graph in unbatch(batch):
                    node_mask = (batch.batch == graph.batch)
                    if (error[node_mask] > self.threshold).any():
                        filtered.append(graph)

        # Train classifier
        self.classifier.train()
        opt = torch.optim.Adam(self.classifier.parameters(), lr=1e-4)
        criterion = nn.BCELoss()
        
        for _ in range(epochs):
            for batch in DataLoader(filtered, batch_size=32, shuffle=True):
                batch = batch.to(self.device)
                preds = self.classifier(batch.x, batch.edge_index, batch.batch)
                loss = criterion(preds.squeeze(), batch.y.float())
                
                opt.zero_grad()
                loss.backward()
                opt.step()

    def predict(self, data):
        """Full two-stage prediction"""
        data = data.to(self.device)
        
        # Stage 1: Anomaly detection
        with torch.no_grad():
            recon = self.autoencoder(data.x, data.edge_index)
            error = (recon - data.x).pow(2).mean(dim=1)
            
            # Process each graph in batch
            preds = []
            for graph in unbatch(data):
                node_mask = (data.batch == graph.batch)
                if (error[node_mask] > self.threshold).any():
                    # Stage 2: Classification
                    prob = self.classifier(graph.x, graph.edge_index, 
                                         graph.batch).item()
                    preds.append(1 if prob > 0.5 else 0)
                else:
                    preds.append(0)
            
            return torch.tensor(preds, device=self.device)
