from torch_geometric.nn import HeteroConv, SAGEConv, Linear, GINConv, GATConv, MessagePassing
from torch_geometric.nn.aggr import LSTMAggregation, GRUAggregation
import torch
import torch.nn.functional as F

class SAGE(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels, normalize=True)
                for edge_type in metadata[1]
            })
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        return x_dict
    
class TemporalSAGE(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, num_layers): 
        super().__init__()
        
        # Get the spatial and temporal edge types
        self.spatial_edge_types = metadata[1][:-4]
        self.temporal_edge_types = metadata[1][-4:]

        # Assert that last edge type is the temporal one (assuming nomeclature from graph builder is consistent)
        assert self.temporal_edge_types == [('ip', 'temporal_connection', 'ip'), ('con', 'temporal_connection_same_src_ip', 'con'),('con', 'temporal_connection_same_dst_ip', 'con'),('con', 'temporal_connection_same_flow', 'con')]

        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            # First step. Apply temporal convolutions (to preserve temporal information)
            temporal_conv = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_channels, normalize=True)
                for edge_type in self.temporal_edge_types
            })
            self.convs.append(temporal_conv)

            # Second step. Apply spatial convolutions
            spatial_conv = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_channels, normalize=True)
                for edge_type in self.spatial_edge_types
            })
            self.convs.append(spatial_conv)
            
    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        return x_dict
    
class LinkClassifier(torch.nn.Module):
    def forward(self, x_dict, edge_label_index):
        edge_preds = {}
        for edge_type in edge_label_index:
            feat_1 = x_dict[edge_type[0]][edge_label_index[edge_type][0]]
            feat_2 = x_dict[edge_type[2]][edge_label_index[edge_type][1]]
            # Apply dot-product
            edge_preds[edge_type] = (feat_1 * feat_2).sum(dim=-1)
        return edge_preds
    
class LinkPredictionModel(torch.nn.Module):
    def __init__(self, gnn):
        super().__init__()
        self.gnn = gnn
        self.classifier = LinkClassifier()

    def forward(self, data):
        embedding = self.gnn(data.x_dict, data.edge_label_index_dict)
        preds = self.classifier(embedding, data.edge_label_index_dict)
        return preds
    
class binary_NIDS_model(torch.nn.Module):
    def __init__(self, gnn, hidden_channels=128, number_of_layers=2, temporal=True):
        super().__init__()
        self.gnn = gnn

        self.mlp = torch.nn.ModuleList()
        for _ in range(number_of_layers):
            linear_layer = Linear(-1, hidden_channels)
            relu_layer = torch.nn.LeakyReLU() 
            self.mlp.append(linear_layer)
            self.mlp.append(relu_layer)

        self.readout = torch.nn.Linear(hidden_channels, 1) # We expect a single output to be optimized with BCEWithLogitsLoss
        self.temporal = temporal

    def forward(self, x_dict, edge_index_dict, current_timestep_flow_indices = None):
        x = self.gnn(x_dict, edge_index_dict)
        if self.temporal:
            x = x['con'][current_timestep_flow_indices]
        else:
            x = x['con']

        for layer in self.mlp:
            x = layer(x)

        return self.readout(x.squeeze())
    
class multiclass_NIDS_model(torch.nn.Module):
    def __init__(self, gnn, num_classes, hidden_channels=128, number_of_layers=2, temporal=True):
        super().__init__()
        self.gnn = gnn
        self.layers = torch.nn.Sequential(
            Linear(-1, hidden_channels),
            torch.nn.LeakyReLU(),
            Linear(hidden_channels, num_classes)
        )
        self.readout = torch.nn.Softmax(dim=1)
        self.readout = torch.nn.Softmax(dim=1)
        self.temporal = temporal

    def forward(self, x_dict, edge_index_dict):
        x = self.gnn(x_dict, edge_index_dict)
        x = x['con']

        x = self.layers(x)
        return self.readout(x)
    
# Build simple MLP model
class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers-1):
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(torch.nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for i in range(self.num_layers):
            x = torch.nn.functional.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x