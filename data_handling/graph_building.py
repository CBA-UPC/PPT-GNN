import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
from pandas import testing as tm
import torch
from tqdm import tqdm
from collections import OrderedDict
import torch_geometric.transforms as T
import math

class GraphBuilder:

    def __init__(self):
        pass

    def time_window_with_flow_duration(self, data, window_size, window_stride):
        """
        Optimized algorithm for creating time windows taking into account flow duration.
        """
        timestamps = data['Timestamp'].values.tolist()
        durations = data['Flow Duration Graph Building'].values.tolist()
        flow_ends = [timestamps[i] + durations[i] for i in range(len(timestamps))]
        
        min_timestamp = min(timestamps)
        max_timestamp = max(flow_ends)
        
        # Calculate the number of windows
        num_windows = int((max_timestamp - min_timestamp - window_size) / window_stride) + 1
        
        windows = [[] for _ in range(num_windows)]  # Initialize empty windows list
        
        # Iterate over flow ends and assign indices to all overlapping windows
        for idx, flow_end in enumerate(flow_ends):
            # Calculate the window indices for the current flow
            start_window_idx = max(0, int((timestamps[idx] - min_timestamp - window_size) / window_stride))
            end_window_idx = max(min(num_windows - 1, int((flow_end - min_timestamp) / window_stride)), start_window_idx)
            
            # Assign the index to all overlapping windows
            for window_idx in range(start_window_idx, end_window_idx + 1):
                windows[window_idx].append(idx)
        
        # Convert the list of windows to an OrderedDict to remove duplicate windows and preserve order
        windows = list(OrderedDict.fromkeys(map(tuple, windows)))
        
        # Convert tuples back to lists
        windows = [list(window) for window in windows]

        # Remove empty lists
        windows = [window for window in windows if window]
        
        return windows
    
    def build_spatio_temporal_pyg_graphs(self, windows_indices_list, attributes, labels, lookback_windows, lookback_flows, include_ports=True, features_list=[], attack_mapping={}, connect_inter_window_flows=True):
        graphs_list = []
        window_indices_to_classify_list = []

        # Initialize positional encoding matrix with embedding size 30 and max length 10000
        pe_matrix = self._get_positional_encoding_matrix(30, 10000)

        for window_end in tqdm(range(lookback_windows, len(windows_indices_list)+lookback_windows, lookback_windows)):

            if window_end >= len(windows_indices_list):
                window_end = len(windows_indices_list)

            # get sublist of windows (each window containing flow indexes)
            window_indices_in_memory = windows_indices_list[(window_end-lookback_windows):window_end]
            window_indices_flattened = list([item for sublist in window_indices_in_memory for item in sublist])

            # window indices to be classified
            window_indices_to_classify_list.append(window_indices_flattened)

            # get flattened and unique list of flow indexes to subset features and labels
            flattened_windows = sorted(list(set([item for sublist in window_indices_in_memory for item in sublist])))

            # Build a PyG graph that has spatially disconnected graphs for each time slice/window and temporal edges to connect these slices
            spatio_temporal_graphs = self._build_spatio_temporal_pyg_graph(attributes.loc[flattened_windows], labels.loc[flattened_windows], window_indices_in_memory, lookback_flows, pe_matrix, include_ports, features_list, attack_mapping, connect_inter_window_flows=connect_inter_window_flows)

            # Assert that the number of flows in the graph matches the number of flows in the window
            assert spatio_temporal_graphs['con'].x.shape[0] == len(window_indices_flattened), "Number of flows in graph does not match the number of flows in the window"

            graphs_list.append(spatio_temporal_graphs)

        return graphs_list, window_indices_to_classify_list

    def build_static_pyg_graphs(self, windows_list, attributes, labels, include_ports=True, features_list=[], attack_mapping=None):
        """
        Builds PyG graphs from the given attributes and labels.
        """
        graphs = []
        for window in tqdm(windows_list):
            # Build a PyG graph for each window
            data = self._build_static_pyg_graph(attributes.filter(items=window, axis=0), labels.loc[window], include_ports, features_list, attack_mapping)
            graphs.append(data)

        return graphs
    
    def get_graph_metadata(self, graphs):
        return graphs[0].metadata()
    
    def _build_spatio_temporal_pyg_graph(self, attributes, labels, window_indices, lookback_flows, pe_matrix, include_ports=True, features_list=[], attack_mapping={}, connect_inter_window_flows=False):
        # Make sure labels are in order and do not have gaps
        start_ip_index_counting = 0
        start_con_index_counting = 0
        ip_nodes_mappings_temporal = {}

        # Initialize lists to store the temporal edge index between the flows of the same Source IPs and Destination IPs
        edge_index_con_temporal_src_per_src_ip_list = []
        edge_index_con_temporal_dst_per_src_ip_list = []
        edge_index_con_temporal_src_per_dst_ip_list = []
        edge_index_con_temporal_dst_per_dst_ip_list = []

        num_ips_per_timeslice = []
        if include_ports:
            num_ports_per_timeslice = []

        if include_ports:
            features_tensors, ip_tensors, port_tensors, src_ip_index_connections_dfs, \
            dst_ip_index_connections_dfs, src_port_index_connections_dfs, \
            dst_port_index_connections_dfs, labels_tensors = self._initialize_empty_with_ports()
        else:
            features_tensors, ip_tensors, src_ip_index_connections_dfs, \
            dst_ip_index_connections_dfs, labels_tensors = self._initialize_empty_without_ports()

        connect_inter_window_flows_source, connect_inter_window_flows_dest = self._get_repeated_flow_connection_src_dst_list(window_indices)

        for idx, window in enumerate(window_indices):

            attributes_slice = attributes.loc[window].reset_index(drop=True)
            labels_slice = labels.loc[window].reset_index(drop=True)

            # Make sure labels are in order and do not have gaps
            tm.assert_index_equal(attributes_slice.index, labels_slice.index)

            # Gather all unique IP addresses and map them to integers.
            ip_nodes, ip_nodes_mapping = self._get_ip_nodes_mapping(attributes_slice, start_ip_index_counting)
            num_ips_per_timeslice.append(len(ip_nodes_mapping))
            for ip in ip_nodes_mapping:
                if ip not in ip_nodes_mappings_temporal:
                    ip_nodes_mappings_temporal[ip] = [ip_nodes_mapping[ip]] # Initialize list of mappings for each IP
                else:
                    ip_nodes_mappings_temporal[ip].append(ip_nodes_mapping[ip]) # Append the mapping to the list if IP already exists

            if include_ports:
                # Rename ports such that they are uniquely identified to their IP
                attributes_slice = self._rename_ports_with_identified_ip(attributes_slice)
                # Gather all unique Port addresses and map them to integers.
                port_nodes, port_nodes_mapping = self._get_port_nodes_mapping(attributes_slice, start_ip_index_counting)
                num_ports_per_timeslice.append(len(port_nodes_mapping))
                
            # map the IP addresses in the attributes_slice
            src_ip_index_connections = pd.Series(attributes_slice['Src IP'].map(ip_nodes_mapping))
            dst_ip_index_connections = pd.Series(attributes_slice['Dst IP'].map(ip_nodes_mapping))

            # Add these mapped IP addresses to the list of IP nodes
            src_ip_index_connections_dfs.append(src_ip_index_connections)
            dst_ip_index_connections_dfs.append(dst_ip_index_connections)

            # Get the flow indices that are part of the same Source IPs
            flow_index_per_src_ip = src_ip_index_connections.groupby(src_ip_index_connections).indices
            # Get the flow indices that are part of the same Destination IPs
            flow_index_per_dst_ip = dst_ip_index_connections.groupby(dst_ip_index_connections).indices

            # Build temporal edge index between the flows of the same Source IPs
            edge_index_con_temporal_src_per_src_ip, edge_index_con_temporal_dst_per_src_ip = self._create_inter_flow_temporal_edges_per_ip(flow_index_per_src_ip, lookback_flows, index_start=start_con_index_counting)
            # Build temporal edge index between the flows of the same Destination IPs
            edge_index_con_temporal_src_per_dst_ip, edge_index_con_temporal_dst_per_dst_ip = self._create_inter_flow_temporal_edges_per_ip(flow_index_per_dst_ip, lookback_flows, index_start=start_con_index_counting)
            # Extend the temporal edge index lists
            edge_index_con_temporal_src_per_src_ip_list.extend(edge_index_con_temporal_src_per_src_ip)
            edge_index_con_temporal_dst_per_src_ip_list.extend(edge_index_con_temporal_dst_per_src_ip)
            edge_index_con_temporal_src_per_dst_ip_list.extend(edge_index_con_temporal_src_per_dst_ip)
            edge_index_con_temporal_dst_per_dst_ip_list.extend(edge_index_con_temporal_dst_per_dst_ip)

            if include_ports:
                # map the port addresses in the attributes_slice
                src_port_index_connections = pd.Series(attributes_slice['Src Port'].map(port_nodes_mapping))
                dst_port_index_connections = pd.Series(attributes_slice['Dst Port'].map(port_nodes_mapping))

                # Add these mapped Port addresses to the list of Port nodes
                src_port_index_connections_dfs.append(src_port_index_connections)
                dst_port_index_connections_dfs.append(dst_port_index_connections)

            # Set Node attributes/features
            if features_list == []:
                pre_positional_embedding_feature_size = 0
                # Get positional encoding for order of features within the window
                if features.shape[0] > pe_matrix.shape[0]:
                    # repeat the positional encoding matrix to match the number of flows in the window
                    features_positional_encoding = pe_matrix.repeat(math.ceil(features.shape[0]/pe_matrix.shape[0]), 1)
                    features_positional_encoding = features_positional_encoding[0:features.shape[0]]
                else:
                    features_positional_encoding = pe_matrix[0:features.shape[0]]
                features = torch.tensor(features_positional_encoding, dtype=torch.float)
            else:
                # Select specified features
                features = attributes_slice.loc[:, features_list]

                # Make sure columns alphabetically sorted!
                features = features[sorted(features.columns)]

                # Convert features torch tensors
                features = torch.tensor(features.values, dtype=torch.float)
                pre_positional_embedding_feature_size = features.shape[1]

                # Get positional encoding for order of features within the window
                if features.shape[0] > pe_matrix.shape[0]:
                    # repeat the positional encoding matrix to match the number of flows in the window
                    features_positional_encoding = pe_matrix.repeat(math.ceil(features.shape[0]/pe_matrix.shape[0]), 1)
                    features_positional_encoding = features_positional_encoding[0:features.shape[0]]
                else:
                    features_positional_encoding = pe_matrix[0:features.shape[0]]
                features = torch.cat((features,features_positional_encoding), dim=1)

            # add features to list
            features_tensors.append(features)

            if idx == len(window_indices)-1:
                amount_of_flows_in_final_window = features.shape[0]

            # Get positional encoding matrix for the current time slice and use it for the IP nodes in the slice and set it as IP attributes instead of dummy features
            pe_matrix_ip = pe_matrix[idx] 
            pe_matrix_ip = pe_matrix_ip.repeat(len(ip_nodes_mapping), 1)
            ip_x = torch.ones(len(ip_nodes),pre_positional_embedding_feature_size)
            ip_x = torch.cat((ip_x,pe_matrix_ip), dim=1)
            ip_tensors.append(ip_x)

            if include_ports:
                # Set dummy features of Port nodes (as ones) with column dimension same as features. This helps certain algorithms like GIN to work
                port_x = torch.ones(len(port_nodes),features.shape[1])
                port_tensors.append(port_x)

            # If multiclass, replace them with the target encoding necessary for training
            if attack_mapping:
                labels_slice = labels_slice.to_numpy()
                labels_slice_torch = torch.tensor(np.array([attack_mapping[attack] for attack in labels_slice]))
            else:
                labels_slice_torch = torch.tensor(labels_slice, dtype=torch.float)
            labels_tensors.append(labels_slice_torch)

            start_ip_index_counting += len(ip_nodes_mapping)
            start_con_index_counting += features.shape[0]

        # Concatenate all ip index connections
        all_src_ip_index_connections = pd.concat(src_ip_index_connections_dfs, axis=0, ignore_index=True)
        all_dst_ip_index_connections = pd.concat(dst_ip_index_connections_dfs, axis=0, ignore_index=True)

        if not include_ports:
            # Build edge index from source to connection and connection to destination
            edge_index_src_con = pd.concat([all_src_ip_index_connections, pd.Series(all_src_ip_index_connections.index, name='Con')], axis=1).values.transpose()
            edge_index_con_dst = pd.concat([pd.Series(all_dst_ip_index_connections.index,name='Con'), all_dst_ip_index_connections], axis=1).values.transpose()

        else:
            # concat all port index connections
            all_src_port_index_connections = pd.concat(src_port_index_connections_dfs, axis=0, ignore_index=True)
            all_dst_port_index_connections = pd.concat(dst_port_index_connections_dfs, axis=0, ignore_index=True)

            # build edge index from source IP to source port and source port to connection and connection to destination port and destination IP
            edge_index_src_ip_src_port = pd.concat([all_src_ip_index_connections, all_src_port_index_connections], axis=1).values.transpose()
            edge_index_src_port_con = pd.concat([all_src_port_index_connections, pd.Series(all_src_port_index_connections.index, name='Con')], axis=1).values.transpose()
            edge_index_con_dst_port = pd.concat([pd.Series(all_dst_port_index_connections.index,name='Con'), all_dst_port_index_connections], axis=1).values.transpose()
            edge_index_dst_port_dst_ip = pd.concat([all_dst_port_index_connections, all_dst_ip_index_connections], axis=1).values.transpose()

        # Initialize the heterograph
        data = HeteroData()

        # Add all nodes to graph
        data['con'].x = torch.cat(features_tensors, dim=0)
        data['ip'].x = torch.cat(ip_tensors, dim=0)
        if include_ports:
            data['port'].x = torch.cat(port_tensors, dim=0)

        # add mask for prediction on flows on graph (only having 1 for the flows under interest to classify)
        mask = torch.zeros(data['con'].x.shape[0], dtype=torch.bool)
        mask[-amount_of_flows_in_final_window:] = True
        data['con'].flow_of_interest = mask
        
        # set labels_slice
        data['con'].y = torch.cat(labels_tensors, dim=0)

        # add edges to graph
        if not include_ports:
            data['ip', 'send', 'con'].edge_index = torch.tensor(edge_index_src_con, dtype=torch.long)
            data['con', 'arrive', 'ip'].edge_index = torch.tensor(edge_index_con_dst, dtype=torch.long)
        else:
            data['ip', 'send', 'port'].edge_index = torch.tensor(edge_index_src_ip_src_port, dtype=torch.long)
            data['port', 'send', 'con'].edge_index = torch.tensor(edge_index_src_port_con, dtype=torch.long)
            data['con', 'arrive', 'port'].edge_index = torch.tensor(edge_index_con_dst_port, dtype=torch.long)
            data['port', 'arrive', 'ip'].edge_index = torch.tensor(edge_index_dst_port_dst_ip, dtype=torch.long)

        # Convert all spatial graphs to undirected graphs
        data = T.ToUndirected()(data)

        # Build temporal edge list to connect the slices (IP in time slice t-window_memory connected to all same IPs in slices from ((t-window_memory) to t)
        temporal_edges_src, temporal_edges_dst = self._create_temporal_src_dst_list(ip_nodes_mappings_temporal)
        edge_index_temporal = pd.concat([pd.Series(temporal_edges_src, name='Src'), pd.Series(temporal_edges_dst, name='Dst')], axis=1).values.transpose()
        if temporal_edges_src == []:
            edge_index_temporal = torch.empty(2, 0, dtype=torch.long)
        data['ip', 'temporal_connection', 'ip'].edge_index = torch.tensor(edge_index_temporal, dtype=torch.long)

        # Build temporal edge list to connect flows of the same IP address (source and dest respectively) within a slice
        if len(edge_index_con_temporal_src_per_src_ip_list) > 0:
            # order temporal edge list according to destination flow index using np.argsort
            edge_index_con_temporal_src_per_src_ip_list, edge_index_con_temporal_dst_per_src_ip_list = self._order_by_dest_idx(edge_index_con_temporal_src_per_src_ip_list, edge_index_con_temporal_dst_per_src_ip_list)
            edge_index_inter_flow_temporal_per_src_ip = torch.tensor([edge_index_con_temporal_src_per_src_ip_list, edge_index_con_temporal_dst_per_src_ip_list], dtype=torch.long)
        else:
            edge_index_inter_flow_temporal_per_src_ip = torch.empty(2, 0, dtype=torch.long)
        if len(edge_index_con_temporal_src_per_dst_ip_list) > 0:
            # order temporal edge list according to destination flow index using np.argsort
            edge_index_con_temporal_src_per_dst_ip_list, edge_index_con_temporal_dst_per_dst_ip_list = self._order_by_dest_idx(edge_index_con_temporal_src_per_dst_ip_list, edge_index_con_temporal_dst_per_dst_ip_list)
            edge_index_inter_flow_temporal_per_dst_ip = torch.tensor([edge_index_con_temporal_src_per_dst_ip_list, edge_index_con_temporal_dst_per_dst_ip_list], dtype=torch.long)
        else:
            edge_index_inter_flow_temporal_per_dst_ip = torch.empty(2, 0, dtype=torch.long)

        data['con', 'temporal_connection_same_src_ip', 'con'].edge_index = edge_index_inter_flow_temporal_per_src_ip
        data['con', 'temporal_connection_same_dst_ip', 'con'].edge_index = edge_index_inter_flow_temporal_per_dst_ip
        if connect_inter_window_flows:
            data['con', 'temporal_connection_same_flow', 'con'].edge_index = torch.tensor([connect_inter_window_flows_source, connect_inter_window_flows_dest], dtype=torch.long)
        
        return data

    def _build_static_pyg_graph(self, attributes, labels, include_ports=True, features_list = [], attack_mapping = {}):
        """
        Builds a PyG graph from the given attributes and labels.
        """
        # Make sure labels are in order and do not have gaps
        tm.assert_index_equal(attributes.index, labels.index)
        traces = attributes.reset_index(drop=True)
        labels = labels.reset_index(drop=True)

        # Gather all unique IP addresses and map them to integers.
        ip_nodes, ip_nodes_mapping = self._get_ip_nodes_mapping(traces)

        if include_ports:
            # Rename ports such that they are uniquely identified to their IP
            traces = self._rename_ports_with_identified_ip(traces)
            # Gather all unique Port addresses and map them to integers.
            port_nodes, port_nodes_mapping = self._get_port_nodes_mapping(traces)
            
        # map the IP addresses in the traces
        src_ip_index_connections = pd.Series(traces['Src IP'].map(ip_nodes_mapping))
        dst_ip_index_connections = pd.Series(traces['Dst IP'].map(ip_nodes_mapping))

        if not include_ports:
            # Build edge index from source to connection and connection to destination
            edge_index_src_con = pd.concat([src_ip_index_connections, pd.Series(src_ip_index_connections.index, name='Con')], axis=1).values.transpose()
            edge_index_con_dst = pd.concat([pd.Series(dst_ip_index_connections.index,name='Con'), dst_ip_index_connections], axis=1).values.transpose()

        else:
            # map the port addresses in the traces
            src_port_index_connections = pd.Series(traces['Src Port'].map(port_nodes_mapping))
            dst_port_index_connections = pd.Series(traces['Dst Port'].map(port_nodes_mapping))

            # build edge index from source IP to source port and source port to connection and connection to destination port and destination IP
            edge_index_src_ip_src_port = pd.concat([src_ip_index_connections, src_port_index_connections], axis=1).values.transpose()
            edge_index_src_port_con = pd.concat([src_port_index_connections, pd.Series(src_port_index_connections.index, name='Con')], axis=1).values.transpose()
            edge_index_con_dst_port = pd.concat([pd.Series(dst_port_index_connections.index,name='Con'), dst_port_index_connections], axis=1).values.transpose()
            edge_index_dst_port_dst_ip = pd.concat([dst_port_index_connections, dst_ip_index_connections], axis=1).values.transpose()

        # Set Node attributes/features
        if features_list == []:
            features = np.ones((traces.shape[0],2)) # Dummy features
        else:
            # Select specified features
            features = traces.loc[:, features_list]
            # Make sure columns alphabetically sorted!
            features = features[sorted(features.columns)]

            # Convert features torch tensors
            features = torch.tensor(features.values, dtype=torch.float)

        # Set dummy features of IP nodes (as ones) with column dimension same as features. This helps certain algorithms like GIN to work
        ip_x = torch.ones(len(ip_nodes),features.shape[1])
        if include_ports:
            # Set dummy features of Port nodes (as ones) with column dimension same as features. This helps certain algorithms like GIN to work
            port_x = torch.ones(len(port_nodes),features.shape[1])

        # If multiclass, replace them with the target encoding necessary for training
        if attack_mapping:
            labels = labels.to_numpy()
            labels_torch = torch.Tensor([attack_mapping[attack] for attack in labels])
        else:
            labels_torch = torch.tensor(labels, dtype=torch.float)
        
        # Convert edge indices to torch tensors
        if not include_ports:
            edge_index_src_con = torch.tensor(edge_index_src_con, dtype=torch.long)
            edge_index_con_dst = torch.tensor(edge_index_con_dst, dtype=torch.long)
        else:
            edge_index_src_ip_src_port = torch.tensor(edge_index_src_ip_src_port, dtype=torch.long)
            edge_index_src_port_con = torch.tensor(edge_index_src_port_con, dtype=torch.long)
            edge_index_con_dst_port = torch.tensor(edge_index_con_dst_port, dtype=torch.long)
            edge_index_dst_port_dst_ip = torch.tensor(edge_index_dst_port_dst_ip, dtype=torch.long)

        # Initialize the heterograph
        data = HeteroData()

        # Add all nodes to graph
        data['con'].x = features
        data['ip'].x = ip_x
        if include_ports:
            data['port'].x = port_x

        # add edges to graph
        if not include_ports:
            data['ip', 'send', 'con'].edge_index = edge_index_src_con
            data['con', 'arrive', 'ip'].edge_index = edge_index_con_dst
        else:
            data['ip', 'send', 'port'].edge_index = edge_index_src_ip_src_port
            data['port', 'send', 'con'].edge_index = edge_index_src_port_con
            data['con', 'arrive', 'port'].edge_index = edge_index_con_dst_port
            data['port', 'arrive', 'ip'].edge_index = edge_index_dst_port_dst_ip

        # set labels
        data['con'].y = labels_torch

        # Convert all spatial graphs to undirected graphs
        data = T.ToUndirected()(data)

        return data
        
    def _initialize_empty_without_ports(self):
        # Initialize lists to store tensors
        features_x_list = []
        ip_x_list = []
        labels_torch_list = []
        src_ip_index_connections_list = []
        dst_ip_index_connections_list = []
        return features_x_list, ip_x_list, src_ip_index_connections_list, dst_ip_index_connections_list, labels_torch_list
    
    def _initialize_empty_with_ports(self):
        # Initialize lists to store tensors
        ip_x_list = []
        port_x_list = []
        features_x_list = []
        labels_torch_list = []
        src_ip_index_connections_list = []
        dst_ip_index_connections_list = []
        src_port_index_connections_list = []
        dst_port_index_connections_list = []
        return features_x_list, ip_x_list, port_x_list, src_ip_index_connections_list, dst_ip_index_connections_list, src_port_index_connections_list, dst_port_index_connections_list, labels_torch_list
    
    def _get_positional_encoding_matrix(self, embedding_dim, max_length):
        # Create a positional encoding for the node features
        pos_enc = torch.zeros(max_length, embedding_dim)
        for pos in range(max_length):
            for i in range(0, embedding_dim, 2):
                pos_enc[pos, i] = math.sin(pos / (10000 ** ((2 * i)/embedding_dim)))
                pos_enc[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/embedding_dim)))
        return pos_enc

    def _get_repeated_flow_connection_src_dst_list(self, windows):
            connection_indexes = {}
            flattened_windows = [item for sublist in windows for item in sublist]
            for idx, con in enumerate(flattened_windows):
                if con not in connection_indexes:
                    connection_indexes[con] = [idx]
                else:
                    connection_indexes[con].append(idx)

            source_indices = []
            target_indices = []
            for con in connection_indexes:
                for i in range(len(connection_indexes[con])-1):
                    source_indices.append(connection_indexes[con][i])
                    target_indices.append(connection_indexes[con][i+1])
            
            return source_indices, target_indices

    def _create_temporal_src_dst_list(self, ip_nodes_mappings_temporal):
        temporal_edges_src = []
        temporal_edges_dst = []
        # Iterate over all IP addresses
        for ip in ip_nodes_mappings_temporal:
            # If there is more than one mapping for the IP, create temporal edges between these
            if len(ip_nodes_mappings_temporal[ip]) > 1:
                # Create all possible temporal edges between the mappings that go forward in time (not backwards)
                for i in range(len(ip_nodes_mappings_temporal[ip])-1):
                    # All indices are sources except the last one. Repeat the source index for the number of destinations
                    temporal_edges_src.extend([ip_nodes_mappings_temporal[ip][i]]*(len(ip_nodes_mappings_temporal[ip])-1-i))
                    # All indices higher than the source index are destinations
                    temporal_edges_dst.extend([ip_nodes_mappings_temporal[ip][i+j+1] for j in range(len(ip_nodes_mappings_temporal[ip])-1-i)])
        
        # Only sort the temporal edges if they are not empty
        if temporal_edges_src:
            temporal_edges_dst, temporal_edges_src = zip(*sorted(zip(temporal_edges_dst, temporal_edges_src)))
        
        return temporal_edges_src, temporal_edges_dst
    
    def _order_by_dest_idx(self, edge_index_con_temporal_src_per_src_ip_list, edge_index_con_temporal_dst_per_src_ip_list):
        # Order the temporal edges by the destination index
        edge_index_con_temporal_src_per_src_ip_list = np.array(edge_index_con_temporal_src_per_src_ip_list)
        edge_index_con_temporal_dst_per_src_ip_list = np.array(edge_index_con_temporal_dst_per_src_ip_list)
        sorted_indices = np.argsort(edge_index_con_temporal_dst_per_src_ip_list)
        edge_index_con_temporal_src_per_src_ip_list = edge_index_con_temporal_src_per_src_ip_list[sorted_indices]
        edge_index_con_temporal_dst_per_src_ip_list = edge_index_con_temporal_dst_per_src_ip_list[sorted_indices]
        
        return edge_index_con_temporal_src_per_src_ip_list, edge_index_con_temporal_dst_per_src_ip_list
    
    def _create_inter_flow_temporal_edges_per_ip(self, flow_indices, lookback_flows, index_start):
        edge_index_con_temporal_src = []
        edge_index_con_temporal_dst = []
        # Iterate over all ip addresses
        for ip in flow_indices:
            # Connect all flows of the same IP address to each other within the lookback window in a temporal manner (forward in time)
            for i in range(len(flow_indices[ip])-1):
                # Connect the source flow to all flows that are within the lookback window
                for j in range(1, min(lookback_flows+1, len(flow_indices[ip])-i)):
                    edge_index_con_temporal_src.append(flow_indices[ip][i]+index_start)
                    edge_index_con_temporal_dst.append(flow_indices[ip][i+j]+index_start)
        
        return edge_index_con_temporal_src, edge_index_con_temporal_dst

    def _rename_ports_with_identified_ip(self, traces):
        traces['Src Port'] = traces['Src IP'].astype(str) + ':' + traces['Src Port'].astype(str)
        traces['Dst Port'] = traces['Dst IP'].astype(str) + ':' + traces['Dst Port'].astype(str)
        return traces

    def _get_port_nodes_mapping(self, traces, start_index=0):
        port_nodes = np.concatenate((traces['Src Port'].unique(), traces['Dst Port'].unique()), axis=0)
        port_nodes = np.unique(port_nodes)
        port_nodes_mapping = dict(zip(port_nodes, range(start_index, start_index+len(port_nodes))))
        return port_nodes, port_nodes_mapping
    
    def _get_ip_nodes_mapping(self, traces, start_index=0):
        ip_nodes = np.concatenate((traces['Src IP'].unique(), traces['Dst IP'].unique()), axis=0)
        ip_nodes = np.unique(ip_nodes)
        ip_nodes_mapping = dict(zip(ip_nodes, range(start_index, start_index+len(ip_nodes))))
        return ip_nodes, ip_nodes_mapping
    
    def _get_temporal_edge_index_per_ip(self, flow_index_per_ip):
        source_indices = []
        target_indices = []
        for ip, flow_indices in flow_index_per_ip.items():
            for i in range(len(flow_indices)-1):
                source_indices.append(flow_indices[i])
                target_indices.append(flow_indices[i+1])

        return torch.tensor([source_indices, target_indices], dtype=torch.long)