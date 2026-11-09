import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from local_code.stage_1_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.mnist_cnn import Method_CNN_MNIST

# Run from the project root (the folder containing local_code/).
loader = Dataset_Loader('MNIST', 'Handwritten digit images')
loader.dataset_source_folder_path = './data/stage_3_data/'
loader.dataset_source_file_name = 'MNIST'
data = loader.load()

model = Method_CNN_MNIST('CNN MNIST', 'Two-layer convolutional classifier')
model.data = data
results = model.run()

true_y = np.asarray(results['true_y'])
pred_y = np.asarray(results['pred_y'])
print('Test accuracy:', accuracy_score(true_y, pred_y))
print('Macro precision:', precision_score(true_y, pred_y, average='macro', zero_division=0))
print('Macro recall:', recall_score(true_y, pred_y, average='macro', zero_division=0))
print('Macro F1:', f1_score(true_y, pred_y, average='macro', zero_division=0))

plt.figure()
plt.plot(model.loss_history, label='Training loss')
plt.xlabel('Epoch')
plt.ylabel('Cross-entropy loss')
plt.legend()
plt.tight_layout()
plt.savefig('mnist_loss.png')

plt.figure()
plt.plot(model.accuracy_history, label='Training accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('mnist_training_accuracy.png')

import torch
torch.save(model.state_dict(), 'mnist_cnn.pth')
