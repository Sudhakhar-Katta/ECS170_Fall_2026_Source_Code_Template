
import pickle

file_path = "result/stage_2_result/test_results.pkl_0"

with open(file_path, "rb") as f:
    results = pickle.load(f)

print("Saved Results:")
print(results)

print("\nEvaluation Metrics:")

for metric, value in results["metrics"].items():
    print(f"{metric}: {value:.4f}")
