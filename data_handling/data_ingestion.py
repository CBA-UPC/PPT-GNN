import random
import numpy as np
import shutil
import json
import pandas as pd
import os

class DataIngestion:

    def __init__(self, raw_data_dir, landed_data_dir, ingested_data_dir, utils_data_dir):
        self.raw_data_dir = raw_data_dir
        self.landed_data_dir = landed_data_dir
        self.utils_data_dir = utils_data_dir
        self.ingested_data_dir = ingested_data_dir
        self._create_directories()
        self._standardize_attack_names()

    def land_raw_NF_UNSW_NB15_files(self, train_size=0.7, val_size=0.15, test_size=0.15):
        
        # Read in the raw data
        data = pd.read_csv(os.path.join(self.raw_data_dir, 'NF_UNSW_NB15', 'nf_unsw_nb15.csv'), header=0)

        # Simple preprocessing steps to be able to use standard methods cross dataset on NF type datasets
        data = self._simple_preprocessing_NF(data)
        data = self._collapse_attacks(data, 'NF_UNSW_NB15')
        
        # Remove large stretch of benign data in the middle of the dataset. Use benign stretch for all train (pre-training) data
        data, benign_stretch = self._remove_large_benign_stretch(data)

        # Because all attacks are at same time, we can split temporally into train, val and test on the continuous benign label. 
        # We wil however not distangle between the different attacks unlike the other datasets
        train, val, test = self._temporal_train_val_test_split(data, 'benign', train_size, val_size, test_size)

        self._write_to_file(train, val, test, 'mixed', 'NF_UNSW_NB15')

        # Write the benign stretch to all train file
        all_train = pd.concat([train, benign_stretch], axis=0)
        all_train.sort_values(by=['Timestamp'], inplace=True)
        all_train.reset_index(drop=True, inplace=True)
        all_train.to_csv(f'{self.landed_data_dir}/NF_UNSW_NB15/all_train_0.csv', index=False)

        # Write counts of all training dataframes to utils
        self._write_all_train_class_counts_to_utils('NF_UNSW_NB15')

    def land_raw_NF_ToN_IoT_files(self, train_size=0.7, val_size=0.15, test_size=0.15):

        data_dir = os.path.join(self.raw_data_dir, 'NF_ToN_IoT')
        all_train_dfs = []

        for datapath in [attack_file for attack_file in os.listdir(data_dir)]:
            # Get the label type from the filename
            label = datapath.split('.')[0]

            if label != 'normal':

                # Read in the raw data
                data = pd.read_csv(os.path.join(data_dir,datapath), header=0)

                # Simple preprocessing steps to be able to use standard methods cross dataset
                data = self._simple_preprocessing_NF(data)
                data, benign_stretch = self._remove_large_benign_stretches(data)
                all_train_dfs.append(benign_stretch)

                # Do first undersampling of the data to make it more manageable, send rest to all train
                if data.shape[0] > 1000000:
                    data, rest_data = self._subsample_to_csv_size(data)
                    all_train_dfs.append(rest_data)

                # Split data into train, val and test while keeping the temporal order
                train, val, test = self._temporal_train_val_test_split(data, label, train_size, val_size, test_size)
                all_train_dfs.append(train)

                # Write the final attack type dataframes to csv files
                self._write_to_file(train, val, test, label, 'NF_ToN_IoT')
            
        # Write all training dataframes to multiple files
        self._write_all_train_to_multiple_files(all_train_dfs, 'NF_ToN_IoT')

        # Write counts of all training dataframes to utils
        self._write_all_train_class_counts_to_utils('NF_ToN_IoT')
    
    def land_raw_NF_BoT_IoT_files(self, train_size=0.7, val_size=0.15, test_size=0.15):

        data_dir = os.path.join(self.raw_data_dir, 'NF_BoT_IoT')
        all_train_dfs = []

        for datapath in [attack_file for attack_file in os.listdir(data_dir)]:
            # Get the label type from the filename
            label = datapath.split('.')[0]

            # Read in the raw data
            data = pd.read_csv(os.path.join(data_dir,datapath), header=0)

            # Simple preprocessing steps to be able to use standard methods cross dataset
            data = self._simple_preprocessing_NF(data, 'subcategory')
            # for ddos, dos, os_scan and service_scan, we'll concat category and subcategory to get the attack type
            if 'dos' in label:
                data['Label'] = data['category'] + '_' + data['Label']
                # Make all attack types lowercase
                data['Label'] = data['Label'].str.lower()
            # rename category to subcategory (for easier use in preprocessing)
            data.rename(columns={'category':'subcategory'}, inplace=True)
            
            data = self._collapse_attacks(data, 'NF_BoT_IoT')
            data, benign_stretch = self._remove_large_benign_stretches(data)
            all_train_dfs.append(benign_stretch)

            # Do first undersampling of the data to make it more manageable, send rest to all train
            if data.shape[0] > 1000000:
                data, rest_data = self._subsample_to_csv_size(data)
                all_train_dfs.append(rest_data)

            # Split data into train, val and test while keeping the temporal order
            train, val, test = self._temporal_train_val_test_split(data, label, train_size, val_size, test_size)
            all_train_dfs.append(train)

            # Write the final attack type dataframes to csv files
            self._write_to_file(train, val, test, label, 'NF_BoT_IoT')
            
        # Write all training dataframes to multiple files
        self._write_all_train_to_multiple_files(all_train_dfs, 'NF_BoT_IoT')

        # Write counts of all training dataframes to utils
        self._write_all_train_class_counts_to_utils('NF_BoT_IoT')

    def ingest_attack_files(self, dataset_name, train_subset_size=50000, val_subset_size=20000, test_subset_size=20000):

        if dataset_name == 'NF_UNSW_NB15':
            print(f' - Dataset {dataset_name} can not be split per attack type, ingesting only mixed and all train files but subsampling a bit...')
            self._ingest_only_mixed(dataset_name, subsample_frac=0.5)

        else:
            print('- Ingesting into binary attack files...')
            
            train_binary_files = [f'{self.landed_data_dir}/{dataset_name}/{data}' for data in os.listdir(f'{self.landed_data_dir}/{dataset_name}') if ('all' not in data and 'train' in data)]
            val_binary_files = [f'{self.landed_data_dir}/{dataset_name}/{data}' for data in os.listdir(f'{self.landed_data_dir}/{dataset_name}') if ('all' not in data and 'val' in data)]
            test_binary_files = [f'{self.landed_data_dir}/{dataset_name}/{data}' for data in os.listdir(f'{self.landed_data_dir}/{dataset_name}') if ('all' not in data and 'test' in data)]

            for file in train_binary_files:
                df = pd.read_csv(file, header=0, index_col=False)
                # Find the indices of the most equal continuous subset of data (no interruptions in time)
                subset_indices = self._find_most_equal_subset_indices(df, train_subset_size)
                df_subset = df.iloc[subset_indices]
                ingested_file_name = f"{self.ingested_data_dir}/{dataset_name}/binary/{file.split('/')[-1]}"
                
                df_subset.to_csv(ingested_file_name, index=False)

            for file in val_binary_files:
                df = pd.read_csv(file, header=0, index_col=False)
                # Find the indices of the most equal continuous subset of data (no interruptions in time)
                subset_indices = self._find_most_equal_subset_indices(df, val_subset_size)
                df_subset = df.iloc[subset_indices]
                ingested_file_name = f"{self.ingested_data_dir}/{dataset_name}/binary/{file.split('/')[-1]}"
                
                df_subset.to_csv(ingested_file_name, index=False)

            for file in test_binary_files:
                df = pd.read_csv(file, header=0, index_col=False)
                # Find the indices of the most equal continuous subset of data (no interruptions in time)
                subset_indices = self._find_most_equal_subset_indices(df, test_subset_size)
                df_subset = df.iloc[subset_indices]
                ingested_file_name = f"{self.ingested_data_dir}/{dataset_name}/binary/{file.split('/')[-1]}"

                df_subset.to_csv(ingested_file_name, index=False)

            # create a proportionate mixed attack file to be used for multi-class classification
            print('- Ingesting binary into proportionate mixed attack files...')
            self._create_mixed_attack_files(dataset_name, train_subset_size, val_subset_size, test_subset_size)

    def ingest_all_train_files(self, dataset_name):
        all_train_files = [f'{self.landed_data_dir}/{dataset_name}/{data}' for data in os.listdir(f'{self.landed_data_dir}/{dataset_name}') if 'all_train' in data]
        for file in all_train_files:
            ingested_file_name = f"{self.ingested_data_dir}/{dataset_name}/pre-training/{file.split('/')[-1]}"
            shutil.copy2(file, ingested_file_name)

    def get_ingested_train_files_and_names(self, dataset_name):
        train_files = [f'{self.ingested_data_dir}/{dataset_name}/binary/{data}' for data in os.listdir(f'{self.ingested_data_dir}/{dataset_name}/binary') if 'train' in data]
        attack_types = [data.split('/')[-1].split('_')[0] for data in train_files]
        return train_files, attack_types
    
    def _balanced_undersampler(self, data, frac, window_stride=1000):
        train_labels = data['Label']

        subsample_size = int(len(train_labels)*frac)
        

        original_data_distribution = train_labels.value_counts()/len(train_labels)

        print(f'Original data distribution ({len(train_labels)} samples):')
        print(original_data_distribution)

        weights = len(train_labels) / train_labels.value_counts()

        best_interval = []
        best_weighted_distance_from_orginial_data = float('inf')

        for start_idx in range(0, len(train_labels), window_stride):
            end_idx = start_idx + subsample_size
            if end_idx > len(train_labels):
                end_idx = len(train_labels)
            window_label_counts = train_labels[start_idx:end_idx].value_counts() / subsample_size
            weighted_distance_from_orginial_data = 0
            for label in window_label_counts.index:
                weighted_distance_from_orginial_data += abs(original_data_distribution[label] - window_label_counts[label]) * weights[label]
            if weighted_distance_from_orginial_data < best_weighted_distance_from_orginial_data:
                best_weighted_distance_from_orginial_data = weighted_distance_from_orginial_data
                best_interval = [start_idx, end_idx]
        
        data = data[best_interval[0]:best_interval[1]]

        data.reset_index(drop=True, inplace=True)

        train_labels = data['Label']

        print(f'Subsampled data distribution ({subsample_size} samples):')
        print(train_labels.value_counts()/len(train_labels))

        return data

    def _create_mixed_attack_files(self, dataset_name, max_amount_per_attack_train, max_amount_per_attack_val, max_amount_per_attack_test, min_size_per_attack_train=4000, min_size_per_attack_val=2000, min_size_per_attack_test=2000):

        # We want similar class distribution as present in all training dataset. More comparable to performance of other models
        class_counts = self.get_all_train_class_counts(dataset_name)
        max_class_count = max(class_counts.values())

        binary_ingested_train_files = [f'{self.ingested_data_dir}/{dataset_name}/binary/{data}' for data in os.listdir(f'{self.ingested_data_dir}/{dataset_name}/binary') if 'train' in data]
        binary_ingested_val_files = [f'{self.ingested_data_dir}/{dataset_name}/binary/{data}' for data in os.listdir(f'{self.ingested_data_dir}/{dataset_name}/binary') if 'val' in data]
        binary_ingested_test_files = [f'{self.ingested_data_dir}/{dataset_name}/binary/{data}' for data in os.listdir(f'{self.ingested_data_dir}/{dataset_name}/binary') if 'test' in data]

        all_train_dfs = []
        for file in binary_ingested_train_files:
            attack = file.split('/')[-1].split('_train.csv')[0]
            df = pd.read_csv(file, header=0, index_col=False)
            subset_frac = class_counts[attack] / max_class_count
            if int(subset_frac*max_amount_per_attack_train) < min_size_per_attack_train:
                all_train_dfs.append(df)
            else:
                subset_indices = self._find_most_equal_subset_indices(df, int(subset_frac*max_amount_per_attack_train), target_fraction=0.9, stride=100)
                all_train_dfs.append(df.iloc[subset_indices])
        ingested_file_name = f"{self.ingested_data_dir}/{dataset_name}/mixed/train.csv"
        final_df = pd.concat(all_train_dfs, axis=0, ignore_index=True)
        final_df.to_csv(ingested_file_name, index=False)

        all_val_dfs = []
        for file in binary_ingested_val_files:
            attack = file.split('/')[-1].split('_val.csv')[0]
            df = pd.read_csv(file, header=0, index_col=False)
            subset_frac = class_counts[attack] / max_class_count
            if int(subset_frac*max_amount_per_attack_val) < min_size_per_attack_val:
                all_val_dfs.append(df)
            else:
                subset_indices = self._find_most_equal_subset_indices(df, int(subset_frac*max_amount_per_attack_val), target_fraction=0.9, stride=100)
                all_val_dfs.append(df.iloc[subset_indices])
        ingested_file_name = f"{self.ingested_data_dir}/{dataset_name}/mixed/val.csv"
        final_df = pd.concat(all_val_dfs, axis=0, ignore_index=True)
        final_df.to_csv(ingested_file_name, index=False)

        all_test_dfs = []
        for file in binary_ingested_test_files:
            attack = file.split('/')[-1].split('_test.csv')[0]
            df = pd.read_csv(file, header=0, index_col=False)
            subset_frac = class_counts[attack] / max_class_count
            if int(subset_frac*max_amount_per_attack_test) < min_size_per_attack_test:
                all_test_dfs.append(df)
            else:
                subset_indices = self._find_most_equal_subset_indices(df, int(subset_frac*max_amount_per_attack_test), target_fraction=0.9, stride=10)
                all_test_dfs.append(df.iloc[subset_indices])
        ingested_file_name = f"{self.ingested_data_dir}/{dataset_name}/mixed/test.csv"
        final_df = pd.concat(all_test_dfs, axis=0, ignore_index=True)
        final_df.to_csv(ingested_file_name, index=False)

    def _find_most_equal_subset_indices(self, df, n, target_fraction=0.5, stride=10000):
        best_frac = 0
        best_idx_start = 0
        if len(df) < n:
            return range(len(df))
        else:
            for idx_start in range(0, int(len(df)-n), stride):
                subset_value_counts = df['Label'].iloc[range(idx_start, idx_start+n)].value_counts()
                if len(subset_value_counts) == 2:
                    frac = subset_value_counts.min() / (subset_value_counts.max()+subset_value_counts.min())
                    if abs(frac - target_fraction) < abs(best_frac - target_fraction):
                        best_frac = frac
                        best_idx_start = idx_start

        return range(best_idx_start, best_idx_start+n)
    
    def _simple_preprocessing_NF(self, data, label_column='category'):
        data[label_column] = data[label_column].fillna('benign')
        data.drop(columns=['classification'], inplace=True)
        data.rename(columns={'bidirectional_first_seen_ms':'Timestamp', label_column: 'Label'}, inplace=True)
        data['Label'] = data['Label'].str.lower()
        data.sort_values(by=['Timestamp'], inplace=True)
        data.reset_index(drop=True, inplace=True)
        return data
    
    def _write_all_train_class_counts_to_utils(self, dataset_name):
        all_train_files = [f'{self.landed_data_dir}/{dataset_name}/{data}' for data in os.listdir(f'{self.landed_data_dir}/{dataset_name}') if 'all_train' in data]
        all_train_dfs = []
        for file in all_train_files:
            df = pd.read_csv(file, header=0, index_col=False)
            all_train_dfs.append(df)
        final_df = pd.concat(all_train_dfs, axis=0, ignore_index=True)
        class_counts = final_df['Label'].value_counts().to_dict()
        with open(f'{self.utils_data_dir}/all_train_class_counts_{dataset_name}.json', 'w') as outfile:
            json.dump(class_counts, outfile)

    def _temporal_train_val_test_split(self, data, attack_type, train_size=0.7, val_size=0.15, test_size=0.15):
        attack_subset = data.loc[(data['Label'] == attack_type)]
        index_list = attack_subset.index # Get index list for all attack rows
        first_row = index_list[0]
        last_row = index_list[-1]
        split_row_test = index_list[round(len(index_list)*train_size)] # Have about a 60/10/30 split in 
        # amount of attack samples in train/test/val
        split_row_val = index_list[round(len(index_list)*(train_size+test_size))]
        train = data.loc[first_row:split_row_test]
        test = data.loc[split_row_test:split_row_val]
        val = data.loc[split_row_val:last_row]
        return train, val, test
    
    def _remove_large_benign_stretch(self, data):
        raw_df_1 = data[data['Timestamp'] < data['Timestamp'].mean()]
        raw_df_2 = data[data['Timestamp'] > data['Timestamp'].mean()]
        timestamp_last_attack_raw_df_1 = raw_df_1.loc[raw_df_1['Label'] != 'benign', 'Timestamp'].max()
        timestamp_first_attack_raw_df_2 = raw_df_2.loc[raw_df_2['Label'] != 'benign', 'Timestamp'].min()
        raw_df_1 = raw_df_1[raw_df_1['Timestamp'] < timestamp_last_attack_raw_df_1]
        raw_df_2 = raw_df_2[raw_df_2['Timestamp'] > timestamp_first_attack_raw_df_2]
        data = pd.concat([raw_df_1, raw_df_2], axis=0)

        benign_stretch_df = data[(data['Timestamp'] > timestamp_last_attack_raw_df_1) & (data['Timestamp'] < timestamp_first_attack_raw_df_2)]

        return data, benign_stretch_df

    def _remove_large_benign_stretches(self, data, max_continuous_frac_allowed=0.1):
        benign_data_pile = []
        gaps_left = True
        while gaps_left:

            benign_or_not = data['Label'] == 'benign'
            count = 0
            max_count = 0
            end_idx = 0
            for idx,item in benign_or_not.iteritems():
                if item:
                    count += 1
                    if count > max_count:
                        end_idx = idx
                        max_count = count
                else:
                    count = 0

            if max_count > max_continuous_frac_allowed*len(data):
                benign_data = data.loc[(end_idx-max_count+1):end_idx]
                benign_data_pile.append(benign_data)

                # Remove indexes in parts for memory reasons
                indexes_to_remove = list(range(end_idx-max_count+1, end_idx+1, max_count//4))
                for idx,idx_start in enumerate(list(range(end_idx-max_count+1, end_idx+1, max_count//4))):
                    idx_end = min((idx_start+max_count//4), end_idx+1)
                    idx_subset_to_remove = list(range(idx_start, idx_end))
                    data.drop(index=idx_subset_to_remove, inplace=True)
                # data = data.drop(benign_data.index)
            else:
                gaps_left = False

        if len(benign_data_pile) > 0:
            benign_data_pile = pd.concat(benign_data_pile, axis=0)
        else:
            benign_data_pile = pd.DataFrame(columns=data.columns)
    
        return data, benign_data_pile
    
    def _ingest_only_mixed(self, dataset_name, subsample_frac=0.5):
        mixed_files = [f'{self.landed_data_dir}/{dataset_name}/{data}' for data in os.listdir(f'{self.landed_data_dir}/{dataset_name}') if 'mixed' in data]
        for file in mixed_files:                
            shutil.copy2(file, f'{self.ingested_data_dir}/{dataset_name}/mixed/{file.split("/")[-1].split("_")[1]}')

    def _collapse_attacks(self, data, dataset_name):
        if dataset_name == 'NF_UNSW_NB15':
            collapse_attacks = {'benign':'benign', 'Exploits': 'exploits', 'Reconnaissance': 'reconnaissance', 'DoS': 'dos', 'Generic': 'generic', 'Shellcode': 'shellcode', 'Fuzzers': 'fuzzers', 'Analysis': 'analysis', 'Worms': 'worms', 'Backdoors': 'backdoor', 'Backdoor':'backdoor'}
        elif dataset_name == 'NF_BoT_IoT':
            collapse_attacks = {'benign':'benign','ddos_tcp': 'ddos_tcp', 'ddos_udp': 'ddos_udp', 'dos_http':'dos_http', 'dos_tcp':'dos_tcp', 'dos_udp':'dos_udp', 'data_exfiltration': 'exfiltration', 'keylogging': 'keylogging', 'os_fingerprint': 'os_scan', 'service_scan': 'service_scan'}
        data['Label'] = data['Label'].map(collapse_attacks)
        return data
    
    def get_all_train_class_counts(self, dataset_name):
        with open(f'{self.utils_data_dir}/all_train_class_counts_{dataset_name}.json', 'r') as infile:
            class_counts = json.load(infile)
        return class_counts
    
    def _subsample_to_csv_size(self, dataframe):
        """
        Script to subsample a dataframe to 1 million rows by taking a random continuous strecth of samples from each million and concatenating them. This way, continuous pieces across the whole dataset are kept.
        """
        rows_in_df = dataframe.shape[0]
        # Amount of millions in the dataframe
        n_of_millions = (rows_in_df // 1000000)
        # Amount of samples to take from each million
        sample_per_million = 1000000//n_of_millions
        # Initialize the subsampled dataframe with the first row
        subsampled_dataframe = pd.DataFrame(columns=dataframe.columns)
        rest_of_data = pd.DataFrame(columns=dataframe.columns)

        for subframe in range(n_of_millions):
            # Randomly select a start and end index for each million of size sample_per_million
            start_idx = random.randint(1000000*subframe, ((1000000*(subframe+1))-sample_per_million))
            end_idx = start_idx + sample_per_million

            # Concatenate the subsampled dataframe with the new subsampled dataframe
            subsampled_dataframe = pd.concat([subsampled_dataframe, dataframe.loc[start_idx:end_idx]], axis=0)

            # get the rest of the data
            rest_of_data = pd.concat([rest_of_data, dataframe.loc[1000000*subframe:start_idx]], axis=0)
            rest_of_data = pd.concat([rest_of_data, dataframe.loc[end_idx:1000000*(subframe+1)]], axis=0)

        return subsampled_dataframe, rest_of_data
    
    def _write_to_file(self, train, val, test, attack_type, dataset_name):
        train.to_csv(f'{self.landed_data_dir}/{dataset_name}/{attack_type}_train.csv', index=False)
        val.to_csv(f'{self.landed_data_dir}/{dataset_name}/{attack_type}_val.csv', index=False)
        test.to_csv(f'{self.landed_data_dir}/{dataset_name}/{attack_type}_test.csv', index=False)

    def _write_all_train_to_multiple_files(self, all_train_dfs, dataset_name):
        all_train_dfs[-1].head(5)
        all_train_df = pd.concat(all_train_dfs, axis=0)
        all_train_df.sort_values(by=['Timestamp'], inplace=True)
        all_train_df.reset_index(drop=True, inplace=True)

        if len(all_train_df.index) < 1000000:
            all_train_df.to_csv(f'{self.landed_data_dir}/{dataset_name}/all_train_0.csv', index=False)

        else:
            file_split_indices = np.arange(0, len(all_train_df.index), 1000000)
            file_split_indices = np.append(file_split_indices, len(all_train_df.index))

            for idx, split_idx in enumerate(file_split_indices[:-1]):
                all_train_df.loc[split_idx:file_split_indices[idx+1]].to_csv(f'{self.landed_data_dir}/{dataset_name}/all_train_{idx}.csv', index=False)
            
    def _create_directories(self):
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.landed_data_dir, exist_ok=True)
        os.makedirs(self.utils_data_dir, exist_ok=True)
        os.makedirs(self.ingested_data_dir, exist_ok=True)
        for dataset in ['NF_UNSW_NB15', 'NF_ToN_IoT', 'NF_BoT_IoT']:
            os.makedirs(os.path.join(self.landed_data_dir, dataset), exist_ok=True)
            for subdir in ['binary', 'mixed', 'pre-training']:
                os.makedirs(os.path.join(self.ingested_data_dir, f'{dataset}/{subdir}'), exist_ok=True)

    def _standardize_attack_names(self):
        # Standardize file names in raw data directory all to lowercase and change all - to _
        for root, dirs, files in os.walk(self.raw_data_dir):
            for file in files:
                os.rename(os.path.join(root, file), os.path.join(root, file.lower()))
                os.rename(os.path.join(root, file), os.path.join(root, file.replace('-', '_')))


