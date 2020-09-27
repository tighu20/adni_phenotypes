import torch
import torch.optim as optim
from torch.nn import BCELoss
from torch.utils.data import DataLoader

from datasets import BrainFeaturesDataset
from models import SimpleMLP


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

def train_simple_mlp():
    train_dataset = BrainFeaturesDataset('data/adni_train_scaled_corrected.csv')
    test_dataset = BrainFeaturesDataset('data/adni_test_scaled_corrected.csv')

    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)

    device = torch.device('cuda:0')

    model = SimpleMLP(dim_in=155).to(device)

    EPOCHS = 100
    LEARNING_RATE = 0.001

    criterion = BCELoss()
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    ##
    # Main cycle
    model.train()
    for e in range(1, EPOCHS + 1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimiser.zero_grad()

            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))

            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        print(
            f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')

if __name__ == '__main__':
    train_simple_mlp()