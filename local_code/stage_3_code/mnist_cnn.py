import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from local_code.base_class.method import method


class Method_CNN_MNIST(method, nn.Module):
    data = None
    max_epoch = 10
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

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.loss_history = []
        self.accuracy_history = []
        self.to(self.device)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

    def train(self, X, y):
        # X is already normalized and channel-first by Dataset_Loader.py.
        X_tensor = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        y_tensor = torch.as_tensor(np.asarray(y), dtype=torch.long)
        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        self.loss_history = []
        self.accuracy_history = []

        nn.Module.train(self, True)
        for epoch in range(self.max_epoch):
            total_loss, correct, total = 0.0, 0, 0
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                logits = self.forward(images)
                loss = loss_function(logits, labels)
                loss.backward()
                optimizer.step()

                n = labels.size(0)
                total_loss += loss.item() * n
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += n

            average_loss = total_loss / total
            train_accuracy = correct / total
            self.loss_history.append(average_loss)
            self.accuracy_history.append(train_accuracy)
            print(f'Epoch {epoch + 1}/{self.max_epoch} | '
                  f'loss: {average_loss:.4f} | train accuracy: {train_accuracy:.4f}')

    def test(self, X):
        nn.Module.train(self, False)
        images = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        loader = DataLoader(images, batch_size=self.batch_size, shuffle=False)
        predictions = []
        with torch.no_grad():
            for batch in loader:
                logits = self.forward(batch.to(self.device))
                predictions.append(logits.argmax(dim=1).cpu())
        return torch.cat(predictions).numpy()

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
