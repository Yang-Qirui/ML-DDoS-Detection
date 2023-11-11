import pandas as pd
import numpy as np
import torch

initial_cols = ['Timestamp', 'Source IP', 'Destination IP']


def process_csv(csv_path, attk_type, cols):
    sorted_sub_dfs = {}
    for df_chunk in pd.read_csv(csv_path, chunksize=5000): # read by chunks, prevent ram exceed
        df_chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_chunk = df_chunk.dropna()
        df_chunk.columns = [col.strip() for col in df_chunk.columns] # remove left and right side space of a column title
        df_chunk = df_chunk[initial_cols + cols] # remove useless columns
        # print(df_chunk.columns)
        filtered = df_chunk[df_chunk['Label'].isin([attk_type, "BENIGN"])].copy() # filter other attacks
        filtered['pd_timestamp'] = pd.to_datetime(filtered['Timestamp']) # convert time
        filtered['ip_pair'] = filtered.apply(lambda row: tuple(sorted([row['Source IP'], row['Destination IP']])), axis=1)
        grouped = filtered.groupby('ip_pair')
        for key, group in grouped:
            if key in sorted_sub_dfs.keys():
                sorted_sub_dfs[key] = pd.concat([sorted_sub_dfs[key], group], axis=0).reset_index(drop=True)
            else:
                sorted_sub_dfs[key] = group

    for k, df in sorted_sub_dfs.items():
        sorted_sub_dfs[k] = df.sort_values(by="pd_timestamp")[cols]
    print(f"{attk_type} finished")
    return sorted_sub_dfs

def min_max_norm(series):
    return (series - series.min()) / (series.max() - series.min())

def load_files_to_tensor(file_name, attack_type, seq_len, train=True):
    '''Here I use 01-12 as trainset, 03-11 as testset'''
    dataset_dir_path = "./dataset/CICDDoS2019"
    path_dict = {
        True:"/".join([dataset_dir_path, "01-12"]),
        False:"/".join([dataset_dir_path, "03-11"])
    }

    feature_cols = ["Destination Port", "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Label"] # features:dst port, byte length [^6] Network Traffic Anomaly Detection Using Recurrent Neural Networks

    csv_path = "/".join([path_dict[train], f"{file_name}.csv"])
    print(f"Start loading {csv_path}, attack type {attack_type}")
    grouped = process_csv(csv_path, attack_type, cols=feature_cols)
    
    flow_feature_ndarrays = []
    for _, df in grouped.items():
        if len(df) == 1:
            continue
        port_series = df['Destination Port'].apply(lambda x: min(x, 10000)) # 10000 from [^6] Network Traffic Anomaly Detection Using Recurrent Neural Networks
        # norm_port_series = min_max_norm(port_series) # cannot use min-max norm, since the port can be the same in a single flow
        byte_series = (df['Total Length of Fwd Packets'] + df['Total Length of Bwd Packets']).apply(lambda x: np.floor(np.log2(x))) # if fwd = bwd = 0, will cause -inf
        byte_series.replace(-np.inf, byte_series[byte_series != -np.inf].min(), inplace=True)
        byte_series.replace(np.inf, byte_series[byte_series != np.inf].max(), inplace=True)
        if np.isnan(byte_series).any():
            continue

        df.loc[df['Label'] == 'BENIGN', 'Label'] = 0
        df.loc[df['Label'] == attack_type, 'Label'] = 1

        feature_ndarray = np.array([port_series.tolist(), byte_series.tolist(), df['Label'].tolist()])
        flow_feature_ndarrays.append(feature_ndarray)

    for i, ndarray in enumerate(flow_feature_ndarrays):
        if seq_len >= ndarray.shape[-1]:
            padding_size = seq_len - ndarray.shape[-1]
        else: 
            padding_size = seq_len - (ndarray.shape[-1] % seq_len)
        padding_conf = [(0, 0)] * ndarray.ndim
        padding_conf[-1] = (0, padding_size)
        padded_ndarray = np.pad(ndarray, pad_width=padding_conf, mode="constant", constant_values=0)
        flow_feature_ndarrays[i] = np.transpose(np.reshape(padded_ndarray, (padded_ndarray.shape[0], seq_len, -1)), (2, 1, 0)) # Batch x seq x features
        
    flow_feature_ndarrays = np.vstack(flow_feature_ndarrays)
    for feature in range(flow_feature_ndarrays.shape[-1] - 1):
        feature_min = flow_feature_ndarrays[..., feature].min(axis=(0, 1), keepdims=True)
        feature_max = flow_feature_ndarrays[..., feature].max(axis=(0, 1), keepdims=True)
        flow_feature_ndarrays[..., feature] = (flow_feature_ndarrays[..., feature] - feature_min) / (feature_max - feature_min)
    # min_vals = np.min(flow_feature_ndarrays[:, :, :-1], axis=-1)
    # max_vals = np.max(flow_feature_ndarrays[:, :, :-1], axis=-1)
    # print(min_vals, max_vals)
    # flow_feature_ndarrays[:, :, :-1] = (flow_feature_ndarrays[:, :, :-1] - min_vals) / (max_vals - min_vals) 
    
    np.save("/".join([path_dict[train], f"{attack_type}.npy"]), flow_feature_ndarrays)
    return torch.from_numpy(flow_feature_ndarrays)

if __name__ == "__main__":
    print("Start reading training set")
    load_files_to_tensor("UDPLag", "UDP-lag", 16)
    load_files_to_tensor("DrDoS_UDP", "DrDoS_UDP", 16)
    load_files_to_tensor("Syn", "Syn", 16)
    # print("Start reading testing set")
    # load_files_to_tensor("UDPLag", "UDP-lag", 16)
    # load_files_to_tensor("DrDoS_UDP", "DrDoS_UDP", 16)
    # load_files_to_tensor("Syn", "Syn", 16)
