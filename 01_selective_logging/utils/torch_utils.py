import torch
from torch.utils.data import Dataset
import rasterio

class SoftDiceLoss(torch.nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        intersection = (inputs * targets).sum()
        dice_coeff = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        loss = 1 - dice_coeff
        return loss
    

class DatasetSamples(Dataset):

    def __init__(self, pathlist: list, transform=None, index=[]) -> None:
        self.pathlist = pathlist  
        self.transform = transform
        self.index = index

    def __getitem__(self, idx):
        path = self.pathlist[idx]

        data = rasterio.open(path)

        data = data.read()[self.index]

        sample, label = data[:,:,:-1], data[:,:,-1:]
 
        if self.transform:
            sample = self.transform(sample)

        return sample, label
        
    def __len__(self):
        return len(self.pathlist)