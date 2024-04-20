import torch
from torch.utils.data import Dataset
import rasterio

import numpy as np
import torch.nn as nn

class CrossEntropyDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyDiceLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, size_average=size_average)
        
    def forward(self, inputs, targets):
        
        # Cross-Entropy Loss
        ce_loss = self.cross_entropy(inputs, targets)
        
        # Dice Loss
        inputs_softmax = torch.softmax(inputs, dim=1)

        inputs_flat = inputs_softmax[:, 0, :, :].contiguous().view(inputs.size(0), -1)
        targets_flat = (targets > 0).float().contiguous().view(targets.size(0),-1)

        #inputs_flat[torch.isnan(inputs_flat)] = 0
        #targets_flat[torch.isnan(targets_flat)] = 0
        
        intersection = torch.sum(inputs_flat * targets_flat, dim=1)
        dice_coeff = (2. * intersection + 1) / (torch.sum(inputs_flat, dim=1) + torch.sum(targets_flat, dim=1) + 1)
        dice_loss = 1 - dice_coeff
        
        # Combined Loss
        # combined_loss = ce_loss + dice_loss.mean()
        combined_loss = dice_loss.mean()

        print('loss', combined_loss)
        
        return combined_loss

    

class DatasetSamples(Dataset):

    def __init__(self, pathlist: list, transform=None, index=[]) -> None:
        self.pathlist = pathlist  
        self.transform = transform
        self.index = index

    def __getitem__(self, idx):
        path = self.pathlist[idx]

        input_data = rasterio.open(path)
        input_image = input_data.read()[self.index]

        sample, label = input_image[:-1,:,:], input_image[-1:,:,:]

        # normalize
        sample = np.stack([self.normalize(sample[x,:,:]) for x in range(0, len(self.index)-1)], axis=0)
        label = np.stack([np.nan_to_num(label[0,:,:])], axis=0)

        sample, label = torch.from_numpy(sample), torch.from_numpy(label)

        sample = torch.nn.functional.interpolate(
            sample.unsqueeze(0),
            size=(512, 512),
            mode='bilinear',
            align_corners=True
        ).squeeze(0)

        label = torch.nn.functional.interpolate(
            label.unsqueeze(0),
            size=(512, 512),
            mode='bilinear',
            align_corners=True
        ).squeeze(0)
     
        #if self.transform:
        #    sample = self.transform(sample)

        return sample, label
        
    def __len__(self):
        return len(self.pathlist)
    

    def normalize(self, data):
        normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
        normalized = np.nan_to_num(normalized)
        return normalized



    
