import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def train_batch(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)

    target = torch.argmax(data['con'].y, dim=1)

    loss = criterion(out, target)
    loss.backward()
    optimizer.step()

    preds = torch.argmax(out.detach(), dim=1)

    return loss.item(), preds.cpu().numpy(), target.detach().cpu().numpy()

def validate_batch(model, data, criterion):
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        target = torch.argmax(data['con'].y, dim=1)

        loss = criterion(out, target)
        preds = torch.argmax(out.detach(), dim=1)

    return loss.item(), preds.cpu().numpy(), target.detach().cpu().numpy()

def calculate_multiclass_metrics(out, target, attack_mapping):
    acc = (out == target).sum()/len(target)
    f1_weighted = f1_score(target, out, average='weighted')
    f1_macro = f1_score(target, out, average='macro')
    print(attack_mapping.keys())
    print(confusion_matrix(target, out))
    return acc, f1_weighted, f1_macro

def calculate_multiclass_test_metrics(out, target, probs):
    # Multiclass metrics
    multiclass_acc = (out == target).sum()/len(target)
    multiclass_f1_weighted = f1_score(target, out, average='weighted')
    multiclass_f1_macro = f1_score(target, out, average='macro')

    multiclass_roc_auc_macro_ovr = roc_auc_score(target, probs, labels=range(10), average='macro',multi_class='ovr')
    multiclass_roc_auc_macro_ovo = roc_auc_score(target, probs, labels=range(10), average='macro',multi_class='ovo')
    multiclass_roc_auc_weighted_ovr = roc_auc_score(target, probs, labels=range(10), average='weighted',multi_class='ovr')
    multiclass_roc_auc_weighted_ovo = roc_auc_score(target, probs, labels=range(10), average='weighted',multi_class='ovo')

    # Binary Macro f1. Where if attack classification is considered as anything else than BENIGN ([1, 0, 0, ..., 0]) it is considered detected
    binary_target = target
    binary_out = []
    for idx, i in enumerate(out):
        if i != 0:
            if binary_target[idx] == 0:
                binary_out.append(i)
            else:
                binary_out.append(binary_target[idx])
        else:
            binary_out.append(0)

    binary_macro_f1 = f1_score(binary_target, binary_out, average='macro')
    binary_weighted_f1 = f1_score(binary_target, binary_out, average='weighted')

    return multiclass_acc, multiclass_f1_weighted, multiclass_f1_macro, multiclass_roc_auc_macro_ovr, multiclass_roc_auc_macro_ovo, multiclass_roc_auc_weighted_ovr, multiclass_roc_auc_weighted_ovo, binary_macro_f1, binary_weighted_f1

def export_pretty_confusion_matrix(targets, preds, attack_mapping, save_path):
    for normalization in ['true', 'pred', 'all', None]:
        cm = confusion_matrix(targets, preds, normalize=normalization)
        cm_df = pd.DataFrame(cm, index=attack_mapping.keys(), columns=attack_mapping.keys())

        # Truncate to 3 decimal places
        cm_df = np.trunc(1000 * cm_df) / 1000

        plt.figure(figsize=(10,7))
        sns.heatmap(cm_df, annot=True, fmt='g')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(save_path + f'_{normalization}.png')

def deduplicate_multiclass_sliding_window_results(preds_target_model, ground_truth_target_model, preds_probabilities, windows):

    flattened_windows = [idx for window in windows for idx in window]
    per_index_pred = {key: 0 for key in flattened_windows}
    per_index_ground_truth = {key: 0 for key in flattened_windows}
    per_index_probs = {key: 0 for key in flattened_windows}

    assert len(flattened_windows) == len(preds_target_model) # Check if the number of predictions is equal to the number of indices in flattened windows
    assert len(flattened_windows) == len(ground_truth_target_model) # Check if the number of ground truth is equal to the number of indices in flattened windows

    for idx, pred in enumerate(preds_target_model):
        per_index_probs[flattened_windows[idx]] = preds_probabilities[idx]

    for idx,pred in enumerate(preds_target_model):
        if pred != 0:
            per_index_pred[flattened_windows[idx]] = pred
            per_index_probs[flattened_windows[idx]] = preds_probabilities[idx]

        per_index_ground_truth[flattened_windows[idx]] = ground_truth_target_model[idx]

    preds = [value for key, value in per_index_pred.items()]
    ground_truth = [value for key, value in per_index_ground_truth.items()]
    probs = [value for key, value in per_index_probs.items()]

    return np.array(preds), np.array(ground_truth), np.array(probs)

def balanced_temporal_undersampler(train_attrs, train_labels, frac):
    if frac != 1:
        final_indices = []

        subsample_size = int(len(train_labels)*frac)
        average_class_size = subsample_size // len(train_labels.value_counts())

        classes = list(train_labels.value_counts().index)

        original_data_distribution = train_labels.value_counts()/len(train_labels)

        minority_classes = list(original_data_distribution[original_data_distribution < 0.03].index)
        majority_classes = list(original_data_distribution[original_data_distribution >= 0.03].index)

        for class_name in minority_classes:
            train_labels_indices = [train_labels[train_labels == class_name].index.min(), train_labels[train_labels == class_name].index.max()]
            class_df = train_labels.loc[train_labels_indices[0]:train_labels_indices[1]]

            max_interval_count = 0
            for i in range(0, len(class_df), 100):
                interval_value_count = class_df[i:min(i+average_class_size, len(class_df))].value_counts()
                if class_name in interval_value_count.index:
                    interval_count = interval_value_count[class_name]
                else:
                    interval_count = 0

                if interval_count > max_interval_count:
                    max_interval_count = interval_count
                    indices_to_add = class_df[i:i+average_class_size].index.min(), class_df[i:i+average_class_size].index.max()

            final_indices.extend(list(train_labels.loc[indices_to_add[0]:indices_to_add[1]].index))

        remaining_samples_count = subsample_size - len(final_indices)
        average_class_size = remaining_samples_count // len(majority_classes)

        for class_name in majority_classes:
            train_labels_indices = [train_labels[train_labels == class_name].index.min(), train_labels[train_labels == class_name].index.max()]
            class_df = train_labels.loc[train_labels_indices[0]:train_labels_indices[1]]

            max_interval_count = 0
            for i in range(0, len(class_df), 1000):
                interval_value_count = class_df[i:min(i+average_class_size, len(class_df))].value_counts()
                if class_name in interval_value_count.index:
                    interval_count = interval_value_count[class_name]
                else:
                    interval_count = 0

                if interval_count > max_interval_count:
                    max_interval_count = interval_count
                    indices_to_add = class_df[i:i+average_class_size].index.min(), class_df[i:i+average_class_size].index.max()

            final_indices.extend(list(train_labels.loc[indices_to_add[0]:indices_to_add[1]].index))

        # Because indices are in temporally sorted order!
        final_indices = list(dict.fromkeys(final_indices))
        final_indices.sort()

        practical_frac = len(final_indices)/len(train_labels)

        print(f'Original Dataset Subsampled in balanced temporal way to {practical_frac} % of the original dataset')

        subs_attrs = train_attrs.loc[final_indices]
        subs_labels = train_labels.loc[final_indices]

        pd.testing.assert_index_equal(subs_attrs.index, subs_labels.index)

        subs_attrs.reset_index(drop=True, inplace=True)
        subs_labels.reset_index(drop=True, inplace=True)

        pd.testing.assert_index_equal(subs_attrs.index, subs_labels.index)

    else:
        subs_attrs = train_attrs
        subs_labels = train_labels
        practical_frac = 1

    return subs_attrs, subs_labels, practical_frac