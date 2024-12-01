import os
import json
import random
import torch
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from tqdm import tqdm

def train_model(model, train_graphs, metadata, gnn_type, learning_rate, batch_size, model_dir, checkpoint_interval, starting_epoch=0, epoch_end=200):

    # Link prediction is a binary classification task so we use BCEWithLogitsLoss
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use GPU if available

    # Initialize dataloaders
    loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)

    # Initialize negative sampling transformation
    transform = T.RandomLinkSplit(
            num_val=0.2,
            num_test=0.2,
            disjoint_train_ratio=0.3,
            neg_sampling_ratio=1.0,
            add_negative_train_samples=True,
            edge_types=metadata[1],
            is_undirected=True
        )

    # Set model to device
    model = model.to(device)

    # Train the model
    for epoch in tqdm(range(starting_epoch, epoch_end)):
        total_train_loss = 0
        total_val_loss = 0
        val_accuracies = []
        epoch_results = {}

        for batch in loader:
            train_data, val_data, test_data = transform(batch)
            train_data = _reorder_temporal_indices_transformered_batch(train_data, gnn_type, metadata)
            val_data = _reorder_temporal_indices_transformered_batch(val_data, gnn_type, metadata)
            total_train_loss += _train_batch_link_pred(model, train_data.to(device), optimizer, criterion)
            del train_data # For memory purposes
            batch_val_loss, batch_val_accuracy = _val_batch_link_pred(model, val_data.to(device), criterion)
            del val_data # For memory purposes
            total_val_loss += batch_val_loss
            val_accuracies.append(batch_val_accuracy)

        # Save the results of the epoch
        epoch_results['train_loss'] = total_train_loss
        epoch_results['val_loss'] = total_val_loss
        epoch_results['val_accuracies'] = np.mean(val_accuracies)

        # Save the model checkpoint
        if epoch % checkpoint_interval == 0:
            _save_checkpoint(model, model_dir, epoch, epoch_results)

        print(f'Epoch {epoch} -- Training Loss: {total_train_loss}, Validation Loss: {total_val_loss}, validation Acc: {np.mean(val_accuracies)}')

def _train_batch_link_pred(model, train_data_dict, optimizer, criterion):
        model.train()
        optimizer.zero_grad()
        preds = model(train_data_dict)
        target = train_data_dict.edge_label_dict
        preds_concat = torch.concat(list(preds.values()))
        target_concat = torch.concat(list(target.values()))
        loss = criterion(preds_concat, target_concat)
        loss.backward()
        optimizer.step()
        return float(loss.item())

def _val_batch_link_pred(model, val_data_dict, criterion):
    model.eval()
    with torch.no_grad():
        preds = model(val_data_dict)
        target = val_data_dict.edge_label_dict
        preds_concat = torch.concat(list(preds.values()))
        target_concat = torch.concat(list(target.values()))
        loss = criterion(preds_concat, target_concat)
        accuracy_preds = (torch.round(torch.sigmoid(preds_concat)) == target_concat)
        accuracy = accuracy_preds.sum()/len(accuracy_preds)
    return float(loss.item()), accuracy.item()

def _save_checkpoint(model, model_dir, epoch, epoch_results):

    # Define paths for saving the model checkpoint
    link_prediction_model_path = f'{model_dir}/checkpoint_{epoch}_linkpredictionmodel.pt'
    checkpoint_results_path = f'{model_dir}/checkpoint_{epoch}_results.json'

    # Save model checkpoint for both gnn base and full model
    link_prediction_weights = model.state_dict()
    torch.save(link_prediction_weights, link_prediction_model_path)

    gnn_weights = _extract_gnn_weights(link_prediction_weights)
    gnn_base_model_path = f'{model_dir}/checkpoint_{epoch}_gnnbase.pt'
    torch.save(gnn_weights, gnn_base_model_path)

    # Save epoch results
    with open(checkpoint_results_path, 'w') as f:
        json.dump(epoch_results, f)

def _extract_gnn_weights(link_prediction_weights):
    link_predictor_keys = list(link_prediction_weights.keys())
    gnn_weights = link_prediction_weights
    for key in link_predictor_keys:
        # For a regular dict
        if 'gnn' in key:
            adapted_key = key.replace("gnn.", "")
            gnn_weights[adapted_key] = gnn_weights.pop(key)

    return gnn_weights

def _reorder_temporal_indices_transformered_batch(batch, gnn_type, metadata):
    if gnn_type == 'temporal':
        temporal_edge_types = metadata[1][-4:]
    elif gnn_type == 'static':
        temporal_edge_types = metadata[1][-1]

    for edge_type in temporal_edge_types:
        reorder_indices_edge_indices = torch.argsort(batch[edge_type].edge_index[1])
        reorder_indices_edge_labels = torch.argsort(batch[edge_type].edge_label_index[1])

        batch[edge_type].edge_index = batch[edge_type].edge_index[:, reorder_indices_edge_indices]
        batch[edge_type].edge_label_index = batch[edge_type].edge_label_index[:, reorder_indices_edge_labels]
        batch[edge_type].edge_label = batch[edge_type].edge_label[reorder_indices_edge_labels]

    return batch

def process_data_and_build_graphs(all_datasets, dataset_name, data_preprocessor, graph_builder,gnn_type, window_size, window_memory, include_port, attack_mapping, on_curated_train=True, flow_memory=0, idx=0):
    if on_curated_train:
        dataset = data_preprocessor.load_mixed_train(dataset_name)
    else:
        dataset = data_preprocessor.load_all_train(dataset_name, idx, 0.5)

    if all_datasets:
        preprocessed_train_attrs, labels = data_preprocessor.preprocess_NF('all', dataset, keep_IPs_and_timestamp=True, binary=False, remove_minority_labels=False, only_attacks=False, scale=True, truncate=True)
    else:
        preprocessed_train_attrs, labels = data_preprocessor.preprocess_NF(dataset_name, dataset, keep_IPs_and_timestamp=True, binary=False, remove_minority_labels=False, only_attacks=False, scale=True, truncate=True)

    features = preprocessed_train_attrs.columns
    features_to_use = [feat for feat in features if feat not in ['Dst IP', 'Dst Port', 'Flow Duration Graph Building', 'Src IP', 'Src Port', 'Timestamp']]

    window_list = graph_builder.time_window_with_flow_duration(preprocessed_train_attrs, window_size, window_size)

    window_list = [window_index for window_index in window_list if len(window_index) < 10000]

    if gnn_type == 'temporal':
        train_graphs, _ = graph_builder.build_spatio_temporal__pyg_graphs(window_list, preprocessed_train_attrs, labels, window_memory, flow_memory, include_port, features_to_use, attack_mapping, True)
    
    elif gnn_type == 'static':
        train_graphs = graph_builder.build_static_pyg_graphs(window_list, preprocessed_train_attrs, labels, include_port, features_to_use, attack_mapping)
    else:
        raise ValueError('GNN type not recognized!')

    return train_graphs

def process_data_and_build_out_of_context_graphs(datasets_to_build, data_preprocessor, graph_builder, gnn_type, window_size, window_memory, include_port, flow_memory=0, idx=0):
    mixed_train_graphs_list = []
    for dataset_name in datasets_to_build:
        dataset = data_preprocessor.load_all_train(dataset_name, idx, 0.5)
        preprocessed_train_attrs, labels = data_preprocessor.preprocess_NF('all', dataset, keep_IPs_and_timestamp=True, binary=False, remove_minority_labels=False, only_attacks=False, scale=True, truncate=True)
        attack_mapping = data_preprocessor.load_attack_mapping(dataset_name)
    
        features = preprocessed_train_attrs.columns
        features_to_use = [feat for feat in features if feat not in ['Dst IP', 'Dst Port', 'Flow Duration Graph Building', 'Src IP', 'Src Port', 'Timestamp']]

        window_list = graph_builder.time_window_with_flow_duration(preprocessed_train_attrs, window_size, window_size)

        window_list = [window_index for window_index in window_list if len(window_index) < 10000]

        if gnn_type == 'temporal':
            train_graphs, _ = graph_builder.build_spatio_temporal_pyg_graphs(window_list, preprocessed_train_attrs, labels, window_memory, flow_memory, include_port, features_to_use, attack_mapping, True)
        elif gnn_type == 'static':
            train_graphs = graph_builder.build_static_pyg_graphs(window_list, preprocessed_train_attrs, labels, include_port, features_to_use, attack_mapping)
        else:
            raise ValueError('GNN type not recognized!')
        
        mixed_train_graphs_list.extend(train_graphs)
    
    # Shuffle the graphs
    random.shuffle(mixed_train_graphs_list)

    return mixed_train_graphs_list

def get_metadata_and_sample_graph(all_datasets, dataset_name, data_preprocessor, graph_builder, idx, gnn_type, window_size, window_memory, include_port, attack_mapping, flow_memory=0):
    all_training_datasets = data_preprocessor.load_all_train(dataset_name, idx, 0.01)
    if all_datasets:
        preprocessed_train_attrs, labels = data_preprocessor.preprocess_NF('all', all_training_datasets, keep_IPs_and_timestamp=True, binary=False, remove_minority_labels=False, only_attacks=False, scale=True, truncate=True)
    else:
        preprocessed_train_attrs, labels = data_preprocessor.preprocess_NF(dataset_name, all_training_datasets, keep_IPs_and_timestamp=True, binary=False, remove_minority_labels=False, only_attacks=False, scale=True, truncate=True)

    features = preprocessed_train_attrs.columns
    features_to_use = [feat for feat in features if feat not in ['Dst IP', 'Dst Port', 'Flow Duration Graph Building', 'Src IP', 'Src Port', 'Timestamp']]

    window_list = graph_builder.time_window_with_flow_duration(preprocessed_train_attrs, window_size, window_size)

    if gnn_type == 'temporal':
        sample_graphs, _ = graph_builder.build_spatio_temporal_pyg_graphs(window_list, preprocessed_train_attrs, labels, window_memory, flow_memory, include_port, features_to_use, attack_mapping, True)
    
    elif gnn_type == 'static':
        sample_graphs = graph_builder.build_static_pyg_graphs(window_list, preprocessed_train_attrs, labels, include_port, features_to_use, attack_mapping)
    else:
        raise ValueError('GNN type not recognized!')

    return sample_graphs[0].metadata(), sample_graphs[0], features_to_use

def get_feature_list(features_parameter, dataset_columns):
    if type(features_parameter) == list:
        return features_parameter
    elif features_parameter == 'all':
        # Remove columns that are for graph building
        return [col for col in dataset_columns if col not in ['Dst IP', 'Dst Port', 'Flow Duration Graph Building', 'Src IP', 'Src Port', 'Timestamp']]
    else:
        return []

def load_link_prediction_model(model, model_dir, checkpoint_number, metadata, gnn_type, sample_graph):
    # Initialize negative sampling transformation
    negative_sampling_transform = T.RandomLinkSplit(
            num_val=0.2,
            num_test=0.2,
            disjoint_train_ratio=0.3,
            neg_sampling_ratio=1.0,
            add_negative_train_samples=True,
            edge_types=metadata[1],
            is_undirected=True
        )
    sample_graph = negative_sampling_transform(sample_graph)[0]
    sample_graph = _reorder_temporal_indices_transformered_batch(sample_graph, gnn_type, metadata)

    # Load model
    model_weight_path = os.path.join(model_dir, f'checkpoint_{checkpoint_number}_linkpredictionmodel.pt')
    with torch.no_grad():  # Initialize lazy modules.
        out = model(sample_graph)
    model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu')))
    print(f'Model from path {model_dir} loaded successfully!')
    return model