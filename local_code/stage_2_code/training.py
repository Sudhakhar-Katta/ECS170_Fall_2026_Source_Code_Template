from local_code.stage_1_code.Dataset_Loader import Dataset_Loader
from local_code.stage_1_code.Method_MLP import Method_MLP
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

loader= Dataset_Loader()

#training
loader.dataset_source_folder_path = (
    r"C:\Users\satya\Downloads\ECS170_Fall_2026_Source_Code_Template\ECS170_Fall_2026_Source_Code_Template\data\stage_2_data"
    + "\\"
)

loader.dataset_source_file_name = "train (1).csv"


train_data=loader.load()



X_train=train_data["X"]
y_train=train_data["y"]


model=Method_MLP("MLP","Digits Classifier")
model.train(X_train,y_train)

output_dir=Path("results/stage_2_results")
output_dir.mkdir(parents=True, exist_ok=True)

torch.save(model.state_dict(), "result/stage_2_result/stage_2_mlp.pth")

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(model.loss_history) + 1),
    model.loss_history,
    label="Training Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Stage 2 MLP Training Convergence")
plt.legend()
plt.grid(True)

plt.savefig(
    output_dir / "training_convergence.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()