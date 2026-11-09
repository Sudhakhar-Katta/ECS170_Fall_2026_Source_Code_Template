
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from local_code.base_class.method import method


class Method_CNN_ORL(method, nn.Module):
    data = None
    max_epoch = 80
    learning_rate = 0.001
    batch_size = 32

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 40)

        self.loss_history = []
        self.accuracy_history = []

        self.to(self.device)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def train(self, X, y):
        images = torch.as_tensor(
            np.asarray(X), dtype=torch.float32
        )

        labels = torch.as_tensor(
            np.asarray(y), dtype=torch.long
        )

        dataset = TensorDataset(images, labels)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )

        loss_function = nn.CrossEntropyLoss()

        self.loss_history = []
        self.accuracy_history = []

        nn.Module.train(self, True)

        for epoch in range(self.max_epoch):
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_images, batch_labels in loader:
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.forward(batch_images)
                loss = loss_function(outputs, batch_labels)

                loss.backward()
                optimizer.step()

                count = batch_labels.size(0)

                total_loss += loss.item() * count
                correct += (
                    outputs.argmax(dim=1) == batch_labels
                ).sum().item()
                total += count

            average_loss = total_loss / total
            train_accuracy = correct / total

            self.loss_history.append(average_loss)
            self.accuracy_history.append(train_accuracy)

            print(
                f"Epoch {epoch + 1}/{self.max_epoch} | "
                f"Loss: {average_loss:.4f} | "
                f"Train Accuracy: {train_accuracy:.4f}"
            )

    def test(self, X):
        nn.Module.train(self, False)

        images = torch.as_tensor(
            np.asarray(X), dtype=torch.float32
        )

        loader = DataLoader(
            images,
            batch_size=self.batch_size,
            shuffle=False
        )

        predictions = []

        with torch.no_grad():
            for batch_images in loader:
                batch_images = batch_images.to(self.device)

                outputs = self.forward(batch_images)
                predicted = outputs.argmax(dim=1)

                predictions.append(predicted.cpu())

        return torch.cat(predictions).numpy()

    def run(self):
        print("method running...")

        print("--start training...")
        self.train(
            self.data["train"]["X"],
            self.data["train"]["y"]
        )

        print("--start testing...")
        pred_y = self.test(self.data["test"]["X"])

        return {
            "pred_y": pred_y,
            "true_y": self.data["test"]["y"]
        }
