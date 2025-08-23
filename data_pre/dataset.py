from torch.utils.data import Dataset

class MyDataset(Dataset):
    # Initialization
    def __init__(self, data, label, label_index, mode='2D'):
        self.data, self.label, self.label_index, self.mode = data, label, label_index, mode

    # Get item
    def __getitem__(self, index):
        if self.mode == '2D':
            return self.data[index, :], self.label[index, :], self.label_index[index, :]
        elif self.mode == '3D':
            return self.data[:, index, :], self.label[:, index, :], self.label_index[:, index, :]

    # Get length
    def __len__(self):
        if self.mode == '2D':
            return self.data.shape[0]
        elif self.mode == '3D':
            return self.data.shape[1]

    def getNumpyLabel(self):
        return self.label.cpu().detach().numpy()

    def getTensorTrain(self):
        return self.data