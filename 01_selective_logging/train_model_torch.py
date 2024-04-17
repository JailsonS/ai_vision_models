import torch
import torchvision.transforms as T
import pandas as pd

from utils.torch_utils import *
from glob import glob
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score, jaccard_score
from ..models import RestnetSegmentation

'''

    Config Session

'''

N_EPCOCHS = 50 

CLASSES = 2

LOSS = SoftDiceLoss()

OPTIMIZER = torch.optim.Nadam

METRICS = {
    'recall': recall_score,
    'f1': f1_score,
    'precision': precision_score,
    'intersect of union': jaccard_score,
    'accuracy': accuracy_score
}

PATH_TRAIN = ''
PATH_LOGFILE = 'PATH_TO_CSV_LOGFILE'

'''

    Helpers Class, Functions

'''
   
def calculate_metrics(outputs, labels):

    predictions = torch.sigmoid(outputs) > 0.5
    predictions = predictions.cpu().numpy().astype(int).flatten()

    labels = labels.cpu().numpy().astype(int).flatten()

    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    loss = LOSS(outputs, labels).item()

    return {'Precision': precision, 'Recall': recall, 'F1': f1, 'Loss': loss}


'''


    Normalization Dataset

    
'''

def normalize(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))


'''

    Training Function

'''

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TRANSFORMS = [
        T.RandomVerticalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(degrees=45),
        normalize
    ]

    # Define transformations
    transformations = T.Compose(TRANSFORMS + [T.ToTensor()])

    # Load dataset
    train_dataset = DatasetSamples(pathlist=PATH_TRAIN, transform=transformations)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


    model = RestnetSegmentation(in_channels=11, num_classes=CLASSES).to(device)

    optimizer = OPTIMIZER(model.parameters())


    train_metrics = {'Epoch': [], 'Precision': [], 'Recall': [], 'F1': [], 'Loss': []}

    for epoch in range(N_EPCOCHS):

        train_loss = train(model, train_loader, LOSS, optimizer, device)
        train_metrics['Epoch'].append(epoch+1)
        train_metrics['Loss'].append(train_loss)

        precision_avg, recall_avg, f1_avg, loss_avg = 0, 0, 0, 0
        num_batches = len(train_loader)

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            batch_metrics = calculate_metrics(outputs, labels)
            precision_avg += batch_metrics['Precision']
            recall_avg += batch_metrics['Recall']
            f1_avg += batch_metrics['F1']
            loss_avg += batch_metrics['Loss']

        precision_avg /= num_batches
        recall_avg /= num_batches
        f1_avg /= num_batches
        loss_avg /= num_batches

        train_metrics['Precision'].append(precision_avg)
        train_metrics['Recall'].append(recall_avg)
        train_metrics['F1'].append(f1_avg)

        print(f'Epoch {epoch+1}/{N_EPCOCHS}, Loss: {train_loss:.4f}, Precision: {precision_avg:.4f}, Recall: {recall_avg:.4f}, F1: {f1_avg:.4f}')

    # save training log to CSV
    df = pd.DataFrame(train_metrics)
    df.to_csv(PATH_LOGFILE, index=False)

if __name__ == "__main__":
    main()