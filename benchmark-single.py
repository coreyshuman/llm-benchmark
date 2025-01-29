import sys
import time
from datetime import datetime
import re
import os
from typing import Optional, Mapping, Any
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.tasks import MMLUTask
from ollamachat import OllamaLLM
from ollama import show as oshow, list as olist

def sanitize_filename(filename):
  """Sanitize model names for valid file name characters"""
  # Remove invalid characters: / \ : * ? " < > |
  return re.sub(r"[\\/*?:'<>|]", "_", filename)

def find_key_by_partial_name(dictionary: Mapping[str, Any], partial_key: str):
  """Search for dictionary value by partial key name"""
  return (value for key, value in dictionary.items() if partial_key in key)

# Check for help flag
if "--help" in sys.argv or "-h" in sys.argv:
  available_models = olist()
  print("Usage: python benchmark-single.py <model_name> [<task1> <task2> ...]")
  print("Available models:")
  model_names = [model.model for model in available_models.models if model.model]
  print(model_names)
  print("Available tasks:")
  task_names = [member.name for member in MMLUTask]
  print(task_names)
  sys.exit(0)
    
# Check if model_name and tasks are provided as command-line arguments
if len(sys.argv) < 2:
  print("Usage: python benchmark-single.py <model_name> [<task1> <task2> ...]")
  print("Try --help for more")
  sys.exit(1)

model_name = sys.argv[1]
task_names = sys.argv[2:]

# Define default task if no tasks are provided
if not task_names:
  task_names = ["COLLEGE_COMPUTER_SCIENCE"]

# Map task names to MMLUTask enum values
tasks = []
for task_name in task_names:
  try:
    task = getattr(MMLUTask, task_name)
    tasks.append(task)
  except AttributeError:
    print(f"Warning: Unknown task '{task_name}', skipping.")

if not tasks:
  print("No valid tasks provided. Available tasks: "
    + ", ".join([e.name for e in MMLUTask]))
  sys.exit(1)

# Initialize benchmark
try:
  benchmark = MMLU(tasks=tasks, n_shots=3)
  ollamallm = OllamaLLM(model_name=model_name)
except Exception as e:
  print(f"Error initializing benchmark or model: {e}")
  sys.exit(1)

# Start timing
start_time = time.time()

# Evaluate the model
results = benchmark.evaluate(ollamallm)

# End timing
end_time = time.time()
runtime = end_time - start_time

# Calculate tokens per second
# todo - can we get accurate number from benchmark result?
total_tokens = sum(len(prediction) for prediction in benchmark.predictions)
tokens_per_second = total_tokens / runtime if runtime != 0 else 0

# Generate timestamp and save path
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = "./results/single"

if not os.path.exists(save_path):
  os.makedirs(save_path)

# Create unique log file names
model_name_sanitized = sanitize_filename(model_name)
summary_filename = f"{save_path}/summary_{timestamp}_{model_name_sanitized}"
detailed_predictions_filename = f"{save_path}/detailed_predictions_{timestamp}_{model_name_sanitized}"

model_info = oshow(model=model_name)
model_details = model_info.details

# Log summary to one file
with open(summary_filename + ".log", "w", encoding = "utf-8") as summary_file:
  summary_file.write(f"Model Name: {model_name}\n")
  summary_file.write(f"  Model\n")
  summary_file.write(f"    architecture       {model_details.family}\n")
  summary_file.write(f"    parameters         {model_details.parameter_size}\n")
  summary_file.write(f"    context length     {find_key_by_partial_name(model_info.modelinfo, "context_length")}\n")
  summary_file.write(f"    embedding length   {find_key_by_partial_name(model_info.modelinfo, "embedding_length")}\n")
  summary_file.write(f"    format             {model_details.format}\n")
  summary_file.write(f"    quantization       {model_details.quantization_level}\n")
  summary_file.write(f"Runtime: {runtime:.2f} seconds\n")
  summary_file.write(f"Tokens per Second: {tokens_per_second:.2f}\n")
  summary_file.write(f"Overall Score: {benchmark.overall_score:.2f}\n")
  summary_file.write("Scores:\n")
  # task_scores is a DataFrame with columns "Task" and "Score"
  for index, row in benchmark.task_scores.iterrows():
    task = row["Task"]
    score_value = row["Score"]
    summary_file.write(f"{task}: {score_value}\n")

# Log detailed predictions to another file
with open(detailed_predictions_filename + ".log", "w", encoding = "utf-8") as detail_file:
  for _, row in benchmark.predictions.iterrows():
    task = row["Task"]
    input_text = row["Input"]
    prediction = row["Prediction"]
    correct = row["Correct"] 
    
    detail_file.write(f"Task: {task}\n")
    detail_file.write(f"Input: {input_text}\n")
    detail_file.write(f"Prediction: {prediction}\n")
    detail_file.write(f"Correct: {correct}\n")
    detail_file.write("-" * 40 + "\n")

# Save the detailed predictions to a CSV file
benchmark.predictions.to_csv(detailed_predictions_filename + ".csv", index=False, encoding="utf-8")

print(f"Logging completed. Check '{summary_filename}' and '{detailed_predictions_filename}' for details.")
print("Overall Score: ", benchmark.overall_score)
print("Task-specific Scores: ", benchmark.task_scores)
