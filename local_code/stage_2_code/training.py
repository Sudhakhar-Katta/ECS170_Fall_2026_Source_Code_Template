from local_code.stage_1_code.Dataset_Loader import Dataset_Loader
from local_code.stage_1_code.Method_MLP import Method_MLP
import numpy as np
import torch
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

print(len(X_train))
print(len(X_train[0]))
print(len(np.unique(y_train))) #outputs class count


model=Method_MLP("MLP","Digits Classifier")
model.train(X_train,y_train)

torch.save(model.state_dict(), "result/stage_2_result/stage_2_mlp.pth")