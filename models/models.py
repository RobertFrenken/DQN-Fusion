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

class Autoencoder(torch.nn.Module):
    """Autoencoder with CAN ID embedding for static graphs"""
    def __init__(self, num_ids, in_channels, hidden_dim=32, heads=4, embedding_dim=8, dropout=0.8):
        super().__init__()
        self.id_embedding = nn.Embedding(num_ids, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        gat_in_dim = embedding_dim + (in_channels - 1)
        # Encoder
        self.enc1 = GATConv(gat_in_dim, hidden_dim, heads=heads)
        self.enc2 = GATConv(hidden_dim * heads, hidden_dim, heads=1)
        # Decoder
        self.dec1 = GATConv(hidden_dim, hidden_dim, heads=heads)
        self.dec2 = GATConv(hidden_dim * heads, in_channels, heads=1)
        self.gat_in_dim = gat_in_dim  # Save for output slicing

    def forward(self, x, edge_index):
        id_emb = self.id_embedding(x[:, 0].long())
        other_feats = x[:, 1:]
        x = torch.cat([id_emb, other_feats], dim=1)
        x = self.enc1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.enc2(x, edge_index).relu()
        x = self.dropout(x)
        x = self.dec1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.dec2(x, edge_index).sigmoid()
        return x  # shape: [num_nodes, in_channels]

if __name__ == '__main__':
    # Knowledge Distillation Scenario
    teacher_model = GATWithJK(in_channels=10, hidden_channels=32, out_channels=1, num_layers=5, heads=8)
    student_model = GATWithJK(in_channels=10, hidden_channels=32, out_channels=1, num_layers=2, heads=4)
    net = GATWithJK(10, 8, 1)

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

    model_characteristics(teacher_model)
    print(teacher_model)
    model_characteristics(student_model)
    print(student_model)