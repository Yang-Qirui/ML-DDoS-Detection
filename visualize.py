import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("./dataset/SimulationData/dataset_sdn.csv")

min_flow_length = 4


def feature_extract(df:pd.DataFrame):
    labels = df['label'].tolist()
    
    pass
df = df.sort_values(by="dt")
grouped_df = df.groupby(by=["src", "dst"])[['pktcount', 'bytecount', 'label']]
scaler = MinMaxScaler()
tensors = []
for k,v in grouped_df:
    pkt_series = v['pktcount'].apply(lambda x: np.floor(np.log2(x)))
    pkt_series.replace(-np.inf, pkt_series[pkt_series != -np.inf].min(), inplace=True)
    pkt_series.replace(np.inf, pkt_series[pkt_series != np.inf].max(), inplace=True)
    v['pktcount'] = pkt_series

    byte_series = v['bytecount'].apply(lambda x: np.floor(np.log2(x)))
    byte_series.replace(-np.inf, byte_series[byte_series != -np.inf].min(), inplace=True)
    byte_series.replace(np.inf, byte_series[byte_series != np.inf].max(), inplace=True)
    v['bytecount'] = byte_series

    v.iloc[:, :2] = scaler.fit_transform(v.iloc[:, :2])
    tensor = v.to_numpy()
    # print(tensor[:5])
    tensor = tensor[:int(len(tensor) - len(tensor) % 5)].reshape(-1, 5, 3) #
    tensors.append(tensor)
    # print(tensor[0])
    # assert 0
tensors = np.vstack(tensors)
print(tensors.shape)
np.save("./dataset/SimulationData/sdn_data.npy", tensors)

# for i, row in df.iterrows():
#     ip_dict[row['flow_id']][row["label"]] += 1
#     ip_dict[row['flow_id']]["tot"] += 1

# count_dict = {}
# for k, v in ip_dict.items():
#     tot = v["tot"]
#     if tot in count_dict.keys():
#         count_dict[tot] += 1
#     else:
#         count_dict[tot] = 1

# ordered  = [(k ,v ) for k,v in count_dict.items()]
# ordered.sort(key=lambda x: x[0])
# print(ordered)
# # print(count_dict.keys())
# # import matplotlib.pyplot as plt
# # plt.bar(list(count_dict.keys()),np.log(list(count_dict.values())))
# # plt.show()


# with open("./b.json", "w") as f:
#     json.dump(count_dict, f)
