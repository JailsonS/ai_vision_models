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

EPOCHS = 50 

CLASSES = 2

FEATURE_INDEX = [

]

LOSS = SoftDiceLoss()

OPTIMIZER = torch.optim.Nadam

BATCH_SIZE = 9

METRICS = {
    'recall': recall_score,
    'f1': f1_score,
    'precision': precision_score,
    'intersect of union': jaccard_score,
    'accuracy': accuracy_score
}

MODEL_NAME = 'resnet_v1'

PATH_TRAIN = ''
PATH_TEST = ''

PATH_LOGFILE_TRAIN = f'01_selective_logging/model/{MODEL_NAME}_train.csv'
PATH_LOGFILE_TEST = f'01_selective_logging/model/{MODEL_NAME}_test.csv'

PATH_MODEL = f'path/to/save/{MODEL_NAME}.pth'

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

    Training, Eval Function

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

def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            predictions = torch.sigmoid(outputs) > 0.5
            predictions = predictions.cpu().numpy().astype(int).flatten()
            labels = labels.cpu().numpy().astype(int).flatten()

            all_predictions.extend(predictions)
            all_labels.extend(labels)

    epoch_loss = running_loss / len(data_loader.dataset)
    metrics = calculate_metrics(torch.tensor(all_predictions), torch.tensor(all_labels))
    return epoch_loss, metrics


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TRANSFORMS = [
        T.RandomVerticalFlip(),
        T.RandomHorizontalFlip(),
        T.RandomRotation(),
        normalize
    ]

    # Define transformations
    transformations = T.Compose(TRANSFORMS + [T.ToTensor()])


    train_dataset = DatasetSamples(pathlist=PATH_TRAIN, transform=transformations, index=FEATURE_INDEX)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = DatasetSamples(pathlist=PATH_TEST, transform=transformations, index=FEATURE_INDEX)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = RestnetSegmentation(in_channels=11, num_classes=CLASSES).to(device)

    optimizer = OPTIMIZER(model.parameters())

    train_metrics = {'Epoch': [], 'Precision': [], 'Recall': [], 'F1': [], 'Loss': []}
    test_metrics = {'Precision': [], 'Recall': [], 'F1': [], 'Loss': []}

    for epoch in range(EPOCHS):

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

        print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Precision: {precision_avg:.4f}, Recall: {recall_avg:.4f}, F1: {f1_avg:.4f}')

        # Evaluate on test set
        test_loss, test_metrics_epoch = evaluate(model, test_loader, LOSS, device)
        test_metrics['Loss'].append(test_loss)
        test_metrics['Precision'].append(test_metrics_epoch['Precision'])
        test_metrics['Recall'].append(test_metrics_epoch['Recall'])
        test_metrics['F1'].append(test_metrics_epoch['F1'])

        print(f'Test Loss: {test_loss:.4f}, Precision: {test_metrics_epoch["Precision"]:.4f}, Recall: {test_metrics_epoch["Recall"]:.4f}, F1: {test_metrics_epoch["F1"]:.4f}')

    # Save trained model
    torch.save(model.state_dict(), PATH_MODEL)

    # save training log to CSV
    df_train = pd.DataFrame(train_metrics)
    df_train.to_csv(PATH_LOGFILE_TRAIN, index=False)

    df_test = pd.DataFrame(test_metrics)
    df_test.to_csv(PATH_LOGFILE_TEST, index=False)


if __name__ == "__main__":
    main()