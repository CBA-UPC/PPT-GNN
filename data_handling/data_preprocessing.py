import pandas as pd
import numpy as np
import json
import os
from pandas import testing as tm
import random

class DataPreprocessor:

    def __init__(self, ingested_data_dir, utils_data_dir) -> None:
        self.data_dir = ingested_data_dir
        self.utils_data_dir = utils_data_dir
        os.makedirs(f'{self.utils_data_dir}', exist_ok=True)
        os.makedirs(f'{self.data_dir}', exist_ok=True)
    
    def load_binary_train(self, attack_type, dataset_name) -> pd.DataFrame:
        print('- Loading in training set...')
        data = pd.read_csv(f'{self.data_dir}/{dataset_name}/binary/{attack_type}_train.csv', header=0, index_col=False)
        data.sort_values(by=['Timestamp'], inplace=True, ignore_index=True)
        return data
    
    def load_binary_val(self, attack_type, dataset_name) -> pd.DataFrame:
        print('- Loading in validation set...')
        """
        Loads testingset as a Pandas dataframe.
        """
        data = pd.read_csv(f'{self.data_dir}/{dataset_name}/binary/{attack_type}_val.csv',
                header=0, index_col=False
        )
        data.sort_values(by=['Timestamp'], inplace=True, ignore_index=True)
        return data

    def load_binary_test(self, attack_type, dataset_name) -> pd.DataFrame:
        print('- Loading in test set...')
        """
        Loads testingset as a Pandas dataframe.
        """
        data = pd.read_csv(f'{self.data_dir}/{dataset_name}/binary/{attack_type}_test.csv',
                header=0, index_col=False
        )
        data.sort_values(by=['Timestamp'], inplace=True, ignore_index=True)
        return data
    
    def load_mixed_train(self, dataset_name) -> pd.DataFrame:
        print('- Loading in mixed training set...')
        data = pd.read_csv(f'{self.data_dir}/{dataset_name}/mixed/train.csv',
            header=0, index_col=False
        )
        data.sort_values(by=['Timestamp'], inplace=True, ignore_index=True)
        return data
    
    def load_mixed_val(self, dataset_name) -> pd.DataFrame:
        print('- Loading in mixed validation set...')
        data = pd.read_csv(f'{self.data_dir}/{dataset_name}/mixed/val.csv',
            header=0, index_col=False
        )
        data.sort_values(by=['Timestamp'], inplace=True, ignore_index=True)
        return data
    
    def load_mixed_test(self, dataset_name) -> pd.DataFrame:
        print('- Loading in mixed test set...')
        data = pd.read_csv(f'{self.data_dir}/{dataset_name}/mixed/test.csv',
            header=0, index_col=False
        )
        data.sort_values(by=['Timestamp'], inplace=True, ignore_index=True)
        return data
    
    def load_all_train(self, dataset_name, idx, subset_frac=1) -> pd.DataFrame:
        print(f'- Loading in all training set file nr {idx}...')
        data = pd.read_csv(f'{self.data_dir}/{dataset_name}/pre-training/all_train_{idx}.csv',
            header=0, index_col=False
        )
        data.sort_values(by=['Timestamp'], inplace=True, ignore_index=True)
        random_subset_idx_start = random.randint(0, int(subset_frac*len(data)))
        random_subset_idx_end = random_subset_idx_start + int(subset_frac*len(data))
        data = data.loc[random_subset_idx_start:random_subset_idx_end]
        return data
    
    def get_all_train_files_and_indices(self, dataset_name) -> list:
        all_files = [f'{self.data_dir}/{dataset_name}/pre-training/{file}' for file in os.listdir(f'{self.data_dir}/{dataset_name}/pre-training') if 'all_train' in file]
        return [(file, int(file.split('_')[-1].split('.')[0])) for file in all_files]
    
    # If dataset='all', apply the reconciled cross-dataset preprocessing steps
    def preprocess_NF(self, dataset_name, dataframe, scale=True, truncate=True, keep_IPs_and_timestamp=True, binary=False, remove_minority_labels=False, only_attacks=False):
        # Hard-coded drop of columns that are not needed for the model
        dataframe.drop(['subcategory'], axis=1, inplace=True)

        # Drop rows with NAs
        dataframe = self._handle_missing_values_before_splitting(dataframe)

        # Convert duration from ms to s
        dataframe['bidirectional_duration'] = self._from_ms_to_s(dataframe['bidirectional_duration_ms'])
        dataframe.drop(['bidirectional_duration_ms'], axis=1, inplace=True)
        dataframe['Timestamp'] = self._from_ms_to_s(dataframe['Timestamp'])

        # Rename column to match other datasets
        dataframe.rename(columns={'bidirectional_duration': 'Flow Duration', 'src_ip': 'Src IP', 'dst_ip': 'Dst IP', 'src_port': 'Src Port', 'dst_port': 'Dst Port'}, inplace=True)

        # Make flow duration copy for graph building
        dataframe['Flow Duration Graph Building'] = dataframe['Flow Duration']  # Save the unscaled duration for graph building

        # Get features and numerical columns, split dataset and assert numerical types
        numerical_cols, ohe_cols, cols_to_drop = self.get_features_and_numerical_columns(keep_IPs_and_timestamp)
        attributes_dataframe, labels = self._split_attributes_and_labels(dataframe)
        attributes_dataframe = self._assert_numerical_cols(attributes_dataframe, numerical_cols)

        # One-hot encode the categorical columns
        if not self._ohe_value_file_exists(dataset_name):
            print('-- Unique values for one-hot encoding columns not found. Finding these values and writing to file...')
            self._write_unique_ohe_object_values_to_file(ohe_cols, dataset_name)
        ohe_values_dict = json.load(open(f'{self.utils_data_dir}/ohe_cols_unique_values_{dataset_name}.json'))
        attributes_dataframe, ohe_col_names = self._encode_ohes(attributes_dataframe, ohe_values_dict)

        # Min-Max scale the numerical columns
        if not self._min_max_scaling_file_exists(dataset_name, truncate):
            print('-- Min-Max scaling ranges not found. Calculating these ranges and writing to file...')
            self._write_min_max_scaling_ranges_to_file(numerical_cols, ohe_col_names, dataset_name, truncate)
        if scale:
            print('-- Min-Max scaling numerical columns...')
            if truncate:
                scaling_ranges = pd.read_csv(f'{self.utils_data_dir}/all_train_ranges_{dataset_name}_truncated.csv', header=0, index_col=0)
            else:
                scaling_ranges = pd.read_csv(f'{self.utils_data_dir}/all_train_ranges_{dataset_name}.csv', header=0, index_col=0)
            attributes_dataframe = self._scale_attributes(attributes_dataframe, scaling_ranges, truncate)

        # Remove minority labels and make target boolean
        if remove_minority_labels:
            attributes_dataframe, labels = self._remove_minority_labels(attributes_dataframe, labels)

        # Filter attacks only
        if binary:
            labels = self._make_target_boolean(labels)

        # Handle missing values after splitting
        attributes_dataframe, labels = self._handle_missing_values_after_splitting(attributes_dataframe, labels)

        # Filter attacks only
        if only_attacks:
            attributes_dataframe, labels = self._filter_attacks_only(attributes_dataframe, labels)

        # Sort the dataframe by timestamp, reset index and sort columns alphabetically
        attributes_dataframe, labels = self._sort_and_reset_splitted_data(attributes_dataframe, labels)
        attributes_dataframe = attributes_dataframe[sorted(attributes_dataframe.columns)]

        # Drop columns that are not needed for the model
        attributes_dataframe.drop(cols_to_drop, axis=1, inplace=True)

        return attributes_dataframe, labels
    
    def get_features_and_numerical_columns(self, keep_IPs_and_timestamp=None) -> tuple[list, list, list]:
        numerical_cols = ['Flow Duration', 'bidirectional_packets', 'bidirectional_bytes',
                'bidirectional_min_packet_size', 'bidirectional_max_packet_size','bidirectional_mean_packet_size', 'bidirectional_mean_packet_iat_ms','bidirectional_cumulative_flags', 
                'src2dst_duration_ms', 'src2dst_packets','src2dst_bytes', 'src2dst_min_packet_size', 'src2dst_max_packet_size',
                'src2dst_mean_packet_size', 'src2dst_mean_packet_iat_ms','src2dst_cumulative_flags', 'dst2src_duration_ms', 
                'dst2src_packets','dst2src_bytes', 'dst2src_min_packet_size', 'dst2src_max_packet_size','dst2src_mean_packet_size', 'dst2src_mean_packet_iat_ms','dst2src_cumulative_flags']

        ohe_cols = ['protocol', 'ip_version']

        cols_to_drop = ['bidirectional_last_seen_ms','src2dst_first_seen_ms','src2dst_last_seen_ms','dst2src_first_seen_ms','dst2src_last_seen_ms']

        if not keep_IPs_and_timestamp:
            cols_to_drop.extend(self._get_graph_building_cols())

        return numerical_cols, ohe_cols, cols_to_drop
    
    def load_attack_mapping(self, dataset_name):

        labels = self.get_all_labels_in_processed_dataset(dataset_name)

        labels.sort()

        # Make sure BENIGN is the first label
        labels_in_order = ['BENIGN'] + [label for label in labels if label != 'BENIGN']

        final_dict = {}
        ohe_base = np.zeros((len(labels_in_order), len(labels_in_order)))
        for idx, attack in enumerate(labels_in_order):
                ohe_base[idx][idx] = 1
                final_dict[attack] = ohe_base[idx]

        return final_dict
    
    def get_all_labels_in_processed_dataset(self, dataset_name):
        if dataset_name != 'NF_UNSW_NB15':
            labels = [files.split('_')[0] for files in os.listdir(f'{self.data_dir}/{dataset_name}/binary') if 'train' in files]
            labels.append('BENIGN')
        else:
            labels = ['BENIGN', 'reconnaissance', 'exploits', 'dos', 'generic','shellcode', 'fuzzers', 'worms', 'backdoor', 'analysis']
        return labels
    
    def get_all_train_files_and_indices(self, dataset_name) -> list:
        all_files = [f'{self.data_dir}/{dataset_name}/pre-training/{file}' for file in os.listdir(f'{self.data_dir}/{dataset_name}/pre-training') if 'all_train' in file]
        return [(file, int(file.split('_')[-1].split('.')[0])) for file in all_files]
    
    def _get_graph_building_cols(self) -> list:
        return ['Src IP', 'Dst IP', 'Src Port','Dst Port','Timestamp', 'Flow Duration Graph Building']
    
    def _assert_numerical_cols(self, data, numerical_cols):
        for col in numerical_cols:
            data[col]= pd.to_numeric(data[col],errors='coerce')
        return data
    
    def _handle_missing_values_before_splitting(self, dataframe) -> pd.DataFrame:
        # Drop the (very few) rows with na
        row_nan_count = dataframe.isnull().any(axis=1).sum()
        if row_nan_count > 0:
            print(f'--- Dropping {row_nan_count.sum()} rows with NaNs before preprocessing:')
            dataframe.dropna(axis=0, inplace=True)
            dataframe.reset_index(drop=True, inplace=True)  
        return dataframe
    
    def _handle_missing_values_after_splitting(self, attributes, labels) -> tuple[pd.DataFrame, pd.Series]:
        row_nan_count = attributes.isnull().any(axis=1).sum()
        if row_nan_count > 0:
            print(f'--- Dropping {row_nan_count.sum()} rows with NaNs after min-max-scaling:')
            na_idxs_bool = attributes.isnull().any(axis=1)
            na_idxs = na_idxs_bool.index[na_idxs_bool].tolist()
            attributes.drop(index=na_idxs, axis=1, inplace=True) 
            labels.drop(index=na_idxs, inplace=True)
        tm.assert_index_equal(attributes.index, labels.index)
        return attributes, labels

    def randomly_drop_benign_flows(self, attributes_dataframe, labels, drop_frac = 0.5, random_seed=1) -> tuple[pd.DataFrame, pd.Series]:
        benign_flows = labels[labels == 'BENIGN']
        random_benign_flows = benign_flows.sample(frac=drop_frac, random_state=random_seed)
        attributes_dataframe.drop(random_benign_flows.index, inplace=True, axis=0)
        labels.drop(random_benign_flows.index, inplace=True, axis=0)
        tm.assert_index_equal(attributes_dataframe.index, labels.index)
        attributes_dataframe, labels = self._sort_and_reset_splitted_data(attributes_dataframe, labels)
        return attributes_dataframe, labels
    
    def _split_attributes_and_labels(self, dataframe) -> tuple[pd.DataFrame, pd.Series]:
        X = dataframe.drop('Label', axis=1)
        y = dataframe['Label']
        tm.assert_index_equal(X.index, y.index)
        return X, y
    
    def _sort_and_reset_splitted_data(self, attributes_dataframe, labels) -> tuple[pd.DataFrame, pd.Series]:

        tm.assert_index_equal(attributes_dataframe.index, labels.index)

        attributes_dataframe.sort_values(by=['Timestamp'], inplace=True, ignore_index=False)
        attributes_indexes = attributes_dataframe.index
        labels = labels.loc[attributes_indexes]

        tm.assert_index_equal(attributes_dataframe.index, labels.index)

        attributes_dataframe.reset_index(drop=True, inplace=True)
        labels.reset_index(drop=True, inplace=True)

        return attributes_dataframe, labels
    
    def _remove_minority_labels(self, attributes_dataframe, labels) -> tuple[pd.DataFrame, pd.Series]:
        # Make binary Labels
        while len(labels.unique()) > 2:
            value_counts = labels.value_counts()
            value_counts = value_counts.to_list()
            value_counts.remove(min(value_counts))
            # If the smallest label makes up less than 1% of the dataset
            if min(labels.value_counts()) <= 0.01*min(value_counts):
                print(f'--- In splitting attack data, minority labels found making up {min(labels.value_counts())} samples. Deleting these rows...')
                minority_label = labels.unique()[np.argmin(labels.value_counts())]
                rows_to_drop = labels[labels == minority_label].index
                attributes_dataframe.drop(rows_to_drop, inplace=True)
                labels.drop(rows_to_drop, inplace=True)
            else:
                raise ValueError(f"Too many minority labels found ({min(labels.value_counts())} samples of type {labels.unique()[np.argmin(labels.value_counts())]}) while splitting for this attack type. Please find other split strategy.")
        return attributes_dataframe, labels
    
    def _make_target_boolean(self, labels) -> pd.Series:
        print(f"--- Making target boolean for {len(labels.unique())} labels found in current file...")
        labels = labels != 'BENIGN' # Attack is True, Benign is False  
        return labels

    def _filter_attacks_only(self, attributes_dataframe, labels) -> tuple[pd.DataFrame, pd.Series]:
        attributes_dataframe = attributes_dataframe.loc[labels[labels == 1].index]
        labels = labels.loc[labels == 1]
        attributes_dataframe.reset_index(drop=True, inplace=True)
        labels.reset_index(drop=True, inplace=True)
        return attributes_dataframe, labels
    
    def _ohe_value_file_exists(self, dataset_name) -> bool:
        return os.path.exists(f'{self.utils_data_dir}/ohe_cols_unique_values_{dataset_name}.json')
    
    def _convert(self, o):
        if isinstance(o, np.int64): return int(o)  
        raise TypeError
    
    def _write_unique_ohe_object_values_to_file(self, ohe_col_list, dataset_name):
        '''
        This function is used to check the unique values for the columns that will be one-hot encoded.
        All files of training dataset will be checked.
        '''
        # Get all unique values for the columns that will be one-hot encoded
        ohe_cols_unique_values = {cols: [] for cols in ohe_col_list}
        
        # load in all datasets all-train files
        if dataset_name != 'all':
            mixed_training_files = [f'{self.data_dir}/{dataset_name}/pre-training/{file}' for file in os.listdir(f'{self.data_dir}/{dataset_name}/pre-training') if 'all_train' in file]
        # else load in specific datasets all-train files
        else:
            mixed_training_files = []
            for dataset in os.listdir(self.data_dir):
                mixed_training_files.extend([f'{self.data_dir}/{dataset}/pre-training/{file}' for file in os.listdir(f'{self.data_dir}/{dataset}/pre-training') if 'all_train' in file])

        for idx, training_file_paths in enumerate(mixed_training_files):
            training_df = pd.read_csv(training_file_paths, header=0)
            for col in ohe_col_list:
                ohe_vals_df = [str(unique_val) for unique_val in training_df[col].unique()]
                ohe_cols_unique_values[col].extend(ohe_vals_df)
                ohe_cols_unique_values[col].append('unseen')
                ohe_cols_unique_values[col] = list(set(ohe_cols_unique_values[col]))

        with open(f'{self.utils_data_dir}/ohe_cols_unique_values_{dataset_name}.json', 'w') as f:
            json.dump(ohe_cols_unique_values, f, default=self._convert)

    def _encode_ohes(self, attributes_dataframe, ohe_unique_values_dict) -> pd.DataFrame:
        """
        One-hot encodes a given attribute in the dataset.
        """
        all_encoded_cols = []
        for attribute, values in ohe_unique_values_dict.items():
            ohe = pd.get_dummies(attributes_dataframe[attribute], prefix=attribute, prefix_sep='_')
            # Because there might be missing values in other datasets in the one-hot encoding, we need to add columns for all possible values
            for value in values:
                # If a ohe-value is not in the dataframe, add it to the columns as zero
                if f'{attribute}_{value}' not in ohe.columns:
                    ohe[f'{attribute}_{value}'] = 0
            attributes_ohe_cols = [ohe[(len(attribute)+1):] for ohe in ohe.columns if attribute == ohe[:len(attribute)]]
            for ohe_column in attributes_ohe_cols:
                if ohe_column not in values:
                    ohe[f'{attribute}_unseen'] = 1
                    ohe.drop(f'{attribute}_{ohe_column}', axis=1, inplace=True)
            
            ordered_cols = [f'{str(attribute)}_{value}' for value in sorted(values)]
            all_encoded_cols.extend(ordered_cols)
            ohe = ohe[ordered_cols]
            attributes_dataframe = pd.concat([attributes_dataframe, ohe], axis=1)
            attributes_dataframe.drop(attribute, inplace=True, axis=1)
        return attributes_dataframe, all_encoded_cols
    
    def _min_max_scaling_file_exists(self, dataset_name, truncate=True) -> bool:
        if truncate:
            return os.path.exists(f'{self.utils_data_dir}/all_train_ranges_{dataset_name}_truncated.csv')
        else:
            return os.path.exists(f'{self.utils_data_dir}/all_train_ranges_{dataset_name}.csv')
    
    def _write_min_max_scaling_ranges_to_file(self, numerical_columns, ohe_cols, dataset_name, truncate=True):

        # load in all datasets all-train files
        if dataset_name != 'all':
            mixed_training_files = [f'{self.data_dir}/{dataset_name}/pre-training/{file}' for file in os.listdir(f'{self.data_dir}/{dataset_name}/pre-training') if 'all_train' in file]
        # else load in specific datasets all-train files
        else:
            mixed_training_files = []
            for dataset in os.listdir(self.data_dir):
                mixed_training_files.extend([f'{self.data_dir}/{dataset}/pre-training/{file}' for file in os.listdir(f'{self.data_dir}/{dataset}/pre-training') if 'all_train' in file])

        min_max = pd.DataFrame(columns=numerical_columns+ohe_cols)

        for training_file_paths in mixed_training_files:
            training_df = pd.read_csv(training_file_paths, header=0)
            training_df['bidirectional_duration'] = self._from_ms_to_s(training_df['bidirectional_duration_ms'])
            training_df.drop(['bidirectional_duration_ms', 'subcategory'], axis=1, inplace=True)
            training_df.rename(columns={'duration': 'Flow Duration', 'bidirectional_duration': 'Flow Duration'}, inplace=True)
            self._assert_numerical_cols(training_df, numerical_columns)
            training_df.dropna(axis = 0, inplace=True)

            if truncate:
              minima = training_df.quantile(0.03, numeric_only=True)
              maxima = training_df.quantile(0.97, numeric_only=True)
    
            else:
              minima = training_df.min(axis=0)
              maxima = training_df.max(axis=0)
            for ohe_col in ohe_cols:
                minima[ohe_col] = 0
                maxima[ohe_col] = 1

            minima = minima[numerical_columns+ohe_cols]
            maxima = maxima[numerical_columns+ohe_cols]

            min_max = pd.concat([min_max, pd.DataFrame([minima, maxima], columns=numerical_columns+ohe_cols)], axis=0)

        minima = min_max.min(axis=0)
        maxima = min_max.max(axis=0)

        scaling_cols = numerical_columns + ohe_cols
        scaling_ranges = pd.DataFrame([minima, maxima], columns=scaling_cols, index=['min', 'max'])

        if truncate:
          scaling_ranges.to_csv(f'{self.utils_data_dir}/all_train_ranges_{dataset_name}_truncated.csv')
        
        else:
          scaling_ranges.to_csv(f'{self.utils_data_dir}/all_train_ranges_{dataset_name}.csv')

    def _scale_attributes(self, attributes_dataframe, scaling_ranges, truncate=True) -> pd.DataFrame:
        # Min-Max scale:
        minima = scaling_ranges.loc['min']
        maxima = scaling_ranges.loc['max']

        for col in scaling_ranges.columns:
            if (maxima[col] - minima[col]) != 0:
                if truncate:
                    attributes_dataframe.loc[:, col].clip(lower=minima[col], upper=maxima[col], inplace=True)
                attributes_dataframe.loc[:, col] = (attributes_dataframe.loc[:, col] - minima[col]) / (maxima[col] - minima[col])

        return attributes_dataframe

    def _from_ms_to_s(self, duration_col):
        # convert flow duration from milisecond to second
        return duration_col.apply(lambda x: x*0.001)
      