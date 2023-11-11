from torch.utils.data import Dataset
import numpy as np
import os


def load_npy(dir_path):
    '''Merge all the data from different attack'''
    loaded_data = []
    for file_name in os.listdir(dir_path):
        if file_name.endswith(".npy"):
            file_path = os.path.join(dir_path, file_name)
            data = np.load(file_path)
            loaded_data.append(data)
    return np.vstack(loaded_data)

class DDoSDataset(Dataset):
    def __init__(self, input_tensor) -> None:
        '''
            Number * seq len * [features, label]
        '''
        super().__init__()
        self.x = input_tensor[:, :, :-1]
        self.y = input_tensor[:, :, -1]

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        assert len(self.x) == len(self.y)
        return len(self.x)