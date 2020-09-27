from typing import Dict
import numpy as np
import torch
import torch.optim as optim
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, average_precision_score

from datasets import BrainFeaturesDataset
from models import SimpleMLP


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def calculate_metrics(labels, pred_prob, pred_binary, loss_value) -> Dict[str, float]:
    return {'loss': loss_value,
            'roc': roc_auc_score(labels, pred_prob),
            'p-r': average_precision_score(labels, pred_prob),
            'acc': accuracy_score(labels, pred_binary),
            'f1': f1_score(labels, pred_binary, zero_division=0),
            'sensitivity': recall_score(labels, pred_binary, zero_division=0),
            'specificity': recall_score(labels, pred_binary, pos_label=0, zero_division=0)}


def model_forward_pass(model, loader, is_train, device, criterion, optimiser=None) -> Dict[str, float]:

    if is_train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0
    # For evaluation
    predictions = []
    labels = []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        if is_train:
            optimiser.zero_grad()
            y_pred = model(X_batch)
            # y_pred: BN X 1, y_batch: BN
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            optimiser.step()
        else:
            with torch.no_grad():
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch.unsqueeze(1))

        epoch_loss += loss.item()

        if not is_train:
            predictions.append(y_pred.squeeze().detach().cpu().numpy())
            labels.append(y_batch.cpu().numpy())

    if not is_train:
        predictions = np.hstack(predictions)
        pred_binary = np.where(predictions > 0.5, 1, 0)
        labels = np.hstack(labels)

        return calculate_metrics(labels, predictions, pred_binary,
                                 loss_value=epoch_loss / len(loader))
    else:
        return {'loss': epoch_loss / len(loader)}

def train_simple_mlp():
    train_dataset = BrainFeaturesDataset('data/adni_train_scaled_corrected.csv')
    val_dataset = BrainFeaturesDataset('data/adni_test_scaled_corrected.csv')

    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=200, shuffle=False)

    device = torch.device('cuda:0')

    model = SimpleMLP(dim_in=155).to(device)

    EPOCHS = 100
    LEARNING_RATE = 0.001


    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = BCELoss()

    ##
    # Main cycle
    for e in range(1, EPOCHS + 1):
        _ = model_forward_pass(model=model, loader=train_loader, is_train=True,
                               device=device, optimiser=optimiser, criterion=criterion)
        train_metrics = model_forward_pass(model=model, loader=train_loader, is_train=False,
                                           device=device, criterion=criterion)
        val_metrics = model_forward_pass(model=model, loader=val_loader, is_train=False,
                                           device=device, criterion=criterion)

        print(f'{e + 0:03}| L: {train_metrics["loss"]:.3f} / {val_metrics["loss"]:.3f}'
              f' | Acc: {train_metrics["acc"]:.2f} / {val_metrics["acc"]:.2f}'
              f' | ROC: {train_metrics["roc"]:.2f} / {val_metrics["roc"]:.2f}'
              f' | P-R: {train_metrics["p-r"]:.2f} / {val_metrics["p-r"]:.2f}')

if __name__ == '__main__':
    train_simple_mlp()