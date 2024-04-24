import sys, os

sys.path.append(os.path.abspath('.'))

import shutil
import torch
import torchvision.transforms as T
import pandas as pd
import numpy as np

from utils.torch_utils import *
from glob import glob
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score, jaccard_score
from models.Resnet import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''

    Config Session

'''

EPOCHS = 1 

CLASSES = 1

FEATURE_INDEX = [
    0, 1, 2,
    10, 11, 12,
    22
]

LOSS = CrossEntropyDiceLoss()
#LOSS = torch.nn.CrossEntropyLoss().to(device)

OPTIMIZER = torch.optim.NAdam

BATCH_SIZE = 9

METRICS = {
    'recall': recall_score,
    'f1': f1_score,
    'precision': precision_score,
    'intersect of union': jaccard_score,
    'accuracy': accuracy_score
}

MODEL_NAME = 'resnet_v1'

PATH_TRAIN = '01_selective_logging/data/train'
PATH_TEST = '01_selective_logging/data/test'

PATH_CKPT = f'01_selective_logging/model/{MODEL_NAME}.pth'
PATH_CKPT_BEST = f'01_selective_logging/model/best_{MODEL_NAME}.pth'
PATH_LOGFILE = f'01_selective_logging/model/log_{MODEL_NAME}.csv'
'''

    Helpers Class, Functions

'''
   
def calculate_metrics(outputs, labels):

    predictions = torch.sigmoid(outputs) > 0.5
    predictions = predictions.cpu().detach().numpy().astype(int)

    labels_np = labels.cpu().detach().numpy().astype(int)

    precision = precision_score(labels_np.flatten(), predictions.flatten(), average=None)
    recall = recall_score(labels_np.flatten(), predictions.flatten(), average=None)
    f1 = f1_score(labels_np.flatten(), predictions.flatten(), average=None)
    
    print(precision, recall, f1)

    loss = LOSS(outputs, labels).item()

    return {'Precision': precision, 'Recall': recall, 'F1': f1, 'Loss': loss}


def save_ckpt(state, is_best, ckpt_path, best_model_path):
    f_path = ckpt_path
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_path
        shutil.copyfile(f_path, best_fpath)



'''


    Normalization Dataset

    
'''

def normalize(data):
    data = data.detach().cpu().numpy()
    normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
    return normalized

def flip_up(data):
    return T.RandomVerticalFlip()(data)

def flip_down(data):
    return T.RandomVerticalFlip()(data)

def rotate(data):
    return T.RandomRotation(degrees=45)(data)


'''

    Training, Eval Function

'''

def main():
    # Set device
    
    TRANSFORMS = [
        flip_up,
        flip_down,
        rotate,
        #normalize
    ]

    # define transformations
    transformations = T.Compose(TRANSFORMS)

    # get list samples
    train_samples = list(glob(f'{PATH_TRAIN}/*'))
    test_samples = list(glob(f'{PATH_TEST}/*'))


    train_dataset = DatasetSamples(pathlist=train_samples, index=FEATURE_INDEX)
    test_dataset = DatasetSamples(pathlist=test_samples, index=FEATURE_INDEX)
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    

    model = ResNetSemanticSegmentation(num_classes=CLASSES, in_channels=len(FEATURE_INDEX)-1).to(device)

    optimizer = OPTIMIZER(model.parameters(), lr=0.001)

    min_losses_val = np.Inf

    for epoch in range(EPOCHS):
        loss_val_list = []  
        loss_train_list = []

        model.train()

        print('----------------------------------------------------------')
        print('start training')

        for images, labels in train_loader:
        
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  

            outputs = model(images)
            


            loss = LOSS(outputs, labels)

           
            loss.backward()
            optimizer.step()
    
            loss_train_list.append(loss.cpu().data)


        model.eval()

        print('-------------------------------------------------------------')
        print('start eval')

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = LOSS(outputs, labels)

                loss_val_list.append(loss.cpu().data)
    

        mean_losses_train = np.asarray(loss_val_list).mean()
        mean_losses_val = np.asarray(loss_train_list).mean()
        
        print('Epoch {} \tTrain Loss {:.6f} \tValidation Loss {:.6f}'.format(epoch, mean_losses_train, mean_losses_val))

        # save loss
        logfile = open(PATH_LOGFILE, 'a')
        logfile.write(f'\n{epoch},{mean_losses_train},{mean_losses_val}')
        logfile.close()

        # create checkpoint
        ckpt = {
            'epoch': epoch,
            'val_loss_min': mean_losses_val,
            'state_dict':model.state_dict(),
            'optimizer':optimizer.state_dict()
        }

        # check best model
        if mean_losses_val < min_losses_val:
            print(f'Validation loss decreased ({mean_losses_val} --> {min_losses_val}). Saving model...')
            save_ckpt(ckpt, True, PATH_CKPT, PATH_CKPT_BEST)
            min_losses_val = mean_losses_val


if __name__ == "__main__":
    main()
