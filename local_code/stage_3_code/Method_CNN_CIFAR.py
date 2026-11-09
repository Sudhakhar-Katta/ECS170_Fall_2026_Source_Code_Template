import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from local_code.base_class.method import method


class Method_CNN_CIFAR(method, nn.Module):
    data = None
    max_epoch = 40
    learning_rate = 0.001
    batch_size = 128

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 10)
        )
        self.loss_history = []
        self.accuracy_history = []
        self.to(self.device)

    def forward(self, x):
        return self.classifier(self.features(x))

    def train(self, X, y):
        # The provided Dataset_Loader already outputs NCHW float images in [0, 1].
        images = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        labels = torch.as_tensor(np.asarray(y), dtype=torch.long)
        loader = DataLoader(
            TensorDataset(images, labels),
            batch_size=self.batch_size,
            shuffle=True
        )

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=0.0001
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epoch
        )
        loss_function = nn.CrossEntropyLoss()
        self.loss_history = []
        self.accuracy_history = []

        nn.Module.train(self, True)
        for epoch in range(self.max_epoch):
            total_loss, correct, total = 0.0, 0, 0
            for batch_images, batch_labels in loader:
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)

                # Training-only augmentation: randomly flip images horizontally.
                flip_mask = torch.rand(
                    batch_images.size(0), 1, 1, 1, device=self.device
                ) < 0.5
                batch_images = torch.where(
                    flip_mask, batch_images.flip(dims=[3]), batch_images
                )

                optimizer.zero_grad()
                logits = self.forward(batch_images)
                loss = loss_function(logits, batch_labels)
                loss.backward()
                optimizer.step()

                count = batch_labels.size(0)
                total_loss += loss.item() * count
                correct += (logits.argmax(dim=1) == batch_labels).sum().item()
                total += count

            scheduler.step()
            avg_loss = total_loss / total
            train_acc = correct / total
            self.loss_history.append(avg_loss)
            self.accuracy_history.append(train_acc)
            print(
                f'Epoch {epoch + 1}/{self.max_epoch} | '
                f'Loss: {avg_loss:.4f} | Train Accuracy: {train_acc:.4f}',
                flush=True
            )

    def test(self, X):
        nn.Module.train(self, False)
        images = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        loader = DataLoader(images, batch_size=self.batch_size, shuffle=False)
        predictions = []
        with torch.no_grad():
            for batch_images in loader:
                logits = self.forward(batch_images.to(self.device))
                predictions.append(logits.argmax(dim=1).cpu())
        return torch.cat(predictions).numpy()

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
