from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets import BrainFeaturesDataset
from models import SimpleMLP


def enable_dropout(m):
    for each_module in m.modules():
        if each_module.__class__.__name__.startswith('Dropout'):
            print('Enabling dropout layer.')
            each_module.train()


def mc_passes(model, loader, n_passes: int, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_predictions = []
    std_predictions = []
    id_predictions = []
    for id_batch, X_batch, _ in loader:
        id_batch, X_batch = id_batch.to(device), X_batch.to(device)

        all_preds = []
        with torch.no_grad():
            for _ in range(n_passes):
                all_preds.append(model(X_batch))

        mean_batch = torch.mean(torch.stack(all_preds), dim=0)
        std_batch = torch.std(torch.stack(all_preds), dim=0)

        mean_predictions.append(mean_batch.squeeze().detach().cpu().numpy())
        std_predictions.append(std_batch.squeeze().detach().cpu().numpy())

        id_predictions.append(id_batch.cpu().numpy())

    return np.hstack(id_predictions), np.hstack(mean_predictions), np.hstack(std_predictions)


def run_on_ukb():
    device = torch.device('cpu')

    model = SimpleMLP(dim_in=155).to(device)

    model.load_state_dict(torch.load('saved_models/simple_mlp.pt'))
    model.eval()

    enable_dropout(model)

    ukb_dataset = BrainFeaturesDataset('data/ukb_scaled_corrected.csv', has_target=False, keep_ids=True)
    ukb_loader = DataLoader(ukb_dataset, batch_size=200, shuffle=False)

    ids, means, stds = mc_passes(model, ukb_loader, 10, device)

    ret_df = pd.DataFrame(list(zip(ids, means, stds)), columns=['ukb_id', 'mean', 'std']).set_index('ukb_id')

    ret_df.to_csv('results/simple_output.csv')


if __name__ == '__main__':
    run_on_ukb()
