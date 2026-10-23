
from local_code.stage_1_code.Dataset_Loader import Dataset_Loader
from local_code.stage_1_code.Method_MLP import Method_MLP
from local_code.stage_1_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_1_code.Result_Saver import Result_Saver

import numpy as np
import torch
from pathlib import Path

# 1. Load test dataset
loader = Dataset_Loader()

loader.dataset_source_folder_path = (
    str(Path("data/stage_2_data").resolve()) + "\\"
)

# Replace with your actual labeled test filename
loader.dataset_source_file_name = "ReadMe (1).docx"

test_data = loader.load()

X_test = test_data["X"]
y_test = test_data["y"]

# 2. Initialize the same architecture
model = Method_MLP("MLP", "Digits Classifier")

# 3. Load trained weights
model_path = Path("result/stage_2_result/stage_2_mlp.pth")

model.load_state_dict(
    torch.load(
        model_path,
        map_location=model.device,
        weights_only=True
    )
)

# 4. Generate predictions
# Your custom train() overrides nn.Module.train(),
# so don't call model.eval() yet.
with torch.no_grad():
    X_tensor = torch.as_tensor(
        np.asarray(X_test),
        dtype=torch.float32,
        device=model.device
    ) / 255.0

    outputs = model(X_tensor)
    predictions = outputs.argmax(dim=1).cpu().numpy()

# 5. Calculate metrics
evaluator = Evaluate_Accuracy("test evaluator", "")

evaluator.data = {
    "true_y": np.asarray(y_test),
    "pred_y": predictions
}

evaluator.evaluate()

# 6. Save evaluation results
output_dir = Path("result/stage_2_result")
output_dir.mkdir(parents=True, exist_ok=True)

saver = Result_Saver()

saver.data = {
    "model": "MLP",
    "evaluation_split": "test",
    "metrics": evaluator.metrics,
    "predictions": predictions
}

saver.fold_count = 0
saver.result_destination_folder_path = str(output_dir) + "/"
saver.result_destination_file_name = "test_results.pkl"

saver.save()

print("Test evaluation completed!")
