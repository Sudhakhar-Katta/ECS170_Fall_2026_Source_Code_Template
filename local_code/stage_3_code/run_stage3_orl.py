from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from local_code.stage_1_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL


# Run from the project root, containing data/ and local_code/.
loader = Dataset_Loader('ORL', 'Face recognition images')
loader.dataset_source_folder_path = './data/stage_3_data/'
loader.dataset_source_file_name = 'ORL'
data = loader.load()

model = Method_CNN_ORL('CNN ORL', 'Two convolutional layers for face recognition')
model.data = data
results = model.run()

true_y = np.asarray(results['true_y'])
pred_y = np.asarray(results['pred_y'])
print('Test accuracy:', accuracy_score(true_y, pred_y))
print('Macro precision:', precision_score(true_y, pred_y, average='macro', zero_division=0))
print('Macro recall:', recall_score(true_y, pred_y, average='macro', zero_division=0))
print('Macro F1:', f1_score(true_y, pred_y, average='macro', zero_division=0))

output_dir = Path('results')
output_dir.mkdir(exist_ok=True)

plt.figure()
plt.plot(range(1, len(model.loss_history) + 1), model.loss_history, label='Training loss')
plt.xlabel('Epoch')
plt.ylabel('Cross-entropy loss')
plt.title('ORL CNN training loss')
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / 'orl_loss.png', dpi=160)
plt.close()

plt.figure()
plt.plot(range(1, len(model.accuracy_history) + 1), model.accuracy_history, label='Training accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('ORL CNN training accuracy')
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / 'orl_training_accuracy.png', dpi=160)
plt.close()

torch.save(model.state_dict(), output_dir / 'orl_cnn.pth')
print('Saved model and plots in results/')
