import argparse
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import wandb
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


def run_inference(dataset_location, dataset_id):
    device = torch.device('cuda')
    run_id = 'tjiagom/adni_phenotypes/2cxy59fk'
    api = wandb.Api()
    best_run = api.run(run_id)

    model = SimpleMLP(dim_in=155, dropout_rate=best_run.config['dropout']).to(device)

    restored_path = wandb.restore('simple_mlp.pt', replace=True, run_path=run_id)
    model.load_state_dict(torch.load(restored_path.name))
    model.eval()

    enable_dropout(model)

    dataset = BrainFeaturesDataset(dataset_location, has_target=False, keep_ids=True)
    loader = DataLoader(dataset, batch_size=200, shuffle=False)

    ids, means, stds = mc_passes(model, loader, 50, device)

    ret_df = pd.DataFrame(list(zip(ids, means, stds)), columns=[f'{dataset_id}_id', 'mean', 'std'])
    ret_df = ret_df.set_index(f'{dataset_id}_id')

    ret_df.to_csv(f'results/latest_output_{dataset_id}.csv')

def parse_args():
    parser = argparse.ArgumentParser(description='ADNI Phenotypes')
    parser.add_argument('--dataset_location',
                        type=str,
                        choices=['data/ukb_scaled_corrected.csv', 'data/nacc_scaled_corrected.csv'],
                        help='The location of the dataset.')

    parser.add_argument('--dataset_id',
                        type=str,
                        choices=['ukb', 'nacc'],
                        help='Small identification of dataset.')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)

    run_inference(dataset_location=args.dataset_location, dataset_id=args.dataset_id)
