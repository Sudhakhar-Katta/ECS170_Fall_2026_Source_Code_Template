"""ECS 170 Stage 3-5: one-change-at-a-time CNN ablation experiments.

Run from the project root:
  py -m local_code.stage_3_code.run_stage3_ablations --dataset mnist
  py -m local_code.stage_3_code.run_stage3_ablations --dataset orl
  py -m local_code.stage_3_code.run_stage3_ablations --dataset cifar
  py -m local_code.stage_3_code.run_stage3_ablations --dataset all

Default runs baseline and ablation at seed 42, to enable fair comparisons.
Append --variant-only to skip the baseline rerun.
"""

import argparse
import csv
import gc
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import nn

from local_code.stage_1_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR

# Some projects named the MNIST file mnist_cnn.py rather than Method_CNN_MNIST.py.
try:
    from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
except ModuleNotFoundError as error:
    if error.name != 'local_code.stage_3_code.Method_CNN_MNIST':
        raise
    from local_code.stage_3_code.mnist_cnn import Method_CNN_MNIST


SEED = 42
OUTPUT_DIR = Path('results') / 'ablations'


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(dataset, variant):
    if dataset == 'mnist':
        model = Method_CNN_MNIST('MNIST CNN', 'MNIST ablation')
        model.max_epoch = 10
        if variant:
            # Only change kernel sizes: 3x3 -> 5x5, with padding adjusted
            # to preserve feature-map sizes and the existing fc1 dimensions.
            model.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
            model.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        setting = '5x5 kernels' if variant else '3x3 kernels'

    elif dataset == 'orl':
        model = Method_CNN_ORL('ORL CNN', 'ORL ablation')
        model.max_epoch = 80  # Match your latest reported ORL experiment
        if variant:
            # Only change hidden dimension: 128 -> 64.
            model.fc1 = nn.Linear(32 * 4 * 4, 64)
            model.fc2 = nn.Linear(64, 40)
        setting = 'hidden size 64' if variant else 'hidden size 128'

    elif dataset == 'cifar':
        model = Method_CNN_CIFAR('CIFAR CNN', 'CIFAR ablation')
        model.max_epoch = 40
        if variant:
            # Only change dropout probability: 0.4 -> 0.2.
            dropout_layers = [m for m in model.classifier.modules()
                              if isinstance(m, nn.Dropout)]
            if len(dropout_layers) != 1:
                raise RuntimeError('Expected one Dropout layer in CIFAR classifier')
            dropout_layers[0].p = 0.2
        setting = 'dropout 0.2' if variant else 'dropout 0.4'

    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    model.to(model.device)  # Replacement layers must be moved to GPU if needed.
    return model, setting


def load_data(dataset):
    names = {'mnist': 'MNIST', 'orl': 'ORL', 'cifar': 'CIFAR'}
    name = names[dataset]
    loader = Dataset_Loader(name, f'{name} image classification')
    loader.dataset_source_folder_path = './data/stage_3_data/'
    loader.dataset_source_file_name = name
    return loader.load()


def run_experiment(dataset, data, variant):
    # Re-seed BEFORE constructing each model and starting its DataLoader.
    set_seed()
    model, setting = make_model(dataset, variant)
    model.data = data
    experiment = 'variant' if variant else 'baseline'
    print(f'\n=== {dataset.upper()} | {experiment}: {setting} | seed={SEED} ===', flush=True)
    results = model.run()

    y_true = np.asarray(results['true_y'])
    y_pred = np.asarray(results['pred_y'])
    row = {
        'dataset': dataset.upper(),
        'experiment': experiment,
        'setting': setting,
        'seed': SEED,
        'epochs': model.max_epoch,
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'final_train_loss': model.loss_history[-1],
        'final_train_accuracy': model.accuracy_history[-1],
    }

    print(f"Test accuracy: {row['accuracy']:.4f}")
    print(f"Macro precision: {row['macro_precision']:.4f}")
    print(f"Macro recall: {row['macro_recall']:.4f}")
    print(f"Macro F1: {row['macro_f1']:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = OUTPUT_DIR / f'{dataset}_{experiment}'

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(model.loss_history) + 1), model.loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Cross-entropy loss')
    plt.title(f'{dataset.upper()} {experiment} training loss')
    plt.tight_layout()
    plt.savefig(str(prefix) + '_loss.png', dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(model.accuracy_history) + 1), model.accuracy_history)
    plt.xlabel('Epoch')
    plt.ylabel('Training accuracy')
    plt.title(f'{dataset.upper()} {experiment} training accuracy')
    plt.tight_layout()
    plt.savefig(str(prefix) + '_accuracy.png', dpi=160)
    plt.close()

    torch.save(model.state_dict(), str(prefix) + '_model.pth')
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['mnist', 'orl', 'cifar', 'all'], required=True)
    parser.add_argument('--variant-only', action='store_true',
                        help='Run just the changed configuration')
    args = parser.parse_args()

    targets = ['mnist', 'orl', 'cifar'] if args.dataset == 'all' else [args.dataset]
    all_rows = []
    for dataset in targets:
        data = load_data(dataset)
        baseline = None
        if not args.variant_only:
            baseline = run_experiment(dataset, data, variant=False)
            all_rows.append(baseline)
        variant = run_experiment(dataset, data, variant=True)
        all_rows.append(variant)
        if baseline:
            difference = (variant['accuracy'] - baseline['accuracy']) * 100
            print(f'\n{dataset.upper()} test accuracy change: {difference:+.2f} percentage points')
        del data
        gc.collect()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f'{args.dataset}_comparison.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'\nSaved graphs, models, and comparison CSV to: {OUTPUT_DIR}')
    print(f'CSV: {csv_path}')


if __name__ == '__main__':
    main()
