from torch.utils.data import Dataset


class ConcatenatedDataset(Dataset):
    def __init__(self, dataset_0: Dataset, dataset_1: Dataset):
        self.dataset_0 = dataset_0
        self.dataset_1 = dataset_1

    def __len__(self):
        return len(self.dataset_0) + len(self.dataset_1)

    def __getitem__(self, item):
        if item < len(self.dataset_0):
            return self.dataset_0[item]
        else:
            return self.dataset_1[item - len(self.dataset_0)]
