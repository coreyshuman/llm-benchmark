import sys
import time
from datetime import datetime
import csv
import re
import os
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.tasks import MMLUTask
from ollamachat import OllamaLLM
from ollama import show as oshow, list as olist

def sanitize_filename(filename):
  """Sanitize model names for valid file name characters"""
  # Remove invalid characters: / \ : * ? " < > |
  return re.sub(r"[\\/*?:'<>|]", "_", filename)
  
# Pre-process task names to MMLUTask enum values
def get_tasks(task_names):
  """Get task enum values list from task names"""
  tasks = []
  for task_name in task_names:
    try:
      task = getattr(MMLUTask, task_name)
      tasks.append(task)
    except AttributeError:
      print(f"Warning: Unknown task '{task_name}', skipping.")
  return tasks

def run_benchmark(model_name, tasks):
  """Run benchmark on a single model"""

  try:
    benchmark = MMLU(tasks=tasks, n_shots=5)
    ollamallm = OllamaLLM(model_name=model_name)
  except Exception as e:
    print(f"Error initializing benchmark or model: {e}")
    return None
  
    # Start timing
  start_time = time.time()

  # Evaluate the model
  benchmark.evaluate(ollamallm)

  # End timing
  end_time = time.time()
  runtime = end_time - start_time

  # Calculate tokens per second
  total_tokens = sum(len(prediction) for prediction in benchmark.predictions)
  tokens_per_second = total_tokens / runtime if runtime != 0 else 0

  # Collect task scores for the tasks that ran
  task_scores = {}
  for _, row in benchmark.task_scores.iterrows():
    task = row["Task"].upper()
    score_value = row["Score"]
    task_scores[task] = score_value   

  return {
    "model_name": model_name,
    "runtime": runtime,
    "tokens_per_second": tokens_per_second,
    "overall_score": benchmark.overall_score,
    "task_scores": task_scores
  }


# Check for help flag
if "--help" in sys.argv or "-h" in sys.argv:
  available_models = olist()
  print("Usage: python benchmark-compare.py [<model1> <model2> ...] --tasks [<task1> <task2> ...]")
  
  print("Available models:")
  model_names = [model.model for model in available_models.models if model.model]
  print(model_names)
  
  print("Available tasks:")
  task_names = [member.name for member in MMLUTask]
  print(task_names)
  
  sys.exit(0)

# Parse command-line arguments
model_names = []
task_names = []
args = sys.argv[1:]

# Check if --all flag is provided to run all tests
if "--all" in args:
  task_names = [task.name for task in MMLUTask]
# Find --tasks argument and split models and tasks
elif "--tasks" in args:
  tasks_index = args.index("--tasks")
  model_names = args[:tasks_index]
  task_names = args[tasks_index + 1:]
# If no "--tasks" argument, assume all args are models
else:
  model_names = args  

# Default task if no tasks are specified
if not task_names:
  task_names = ["COLLEGE_COMPUTER_SCIENCE"]
    
# Pre-process task names to MMLUTask enum values globally
tasks = get_tasks(task_names)

if not tasks:
  print("No valid tasks provided. Available tasks: "
    + ", ".join([e.name for e in MMLUTask]))
  sys.exit(1)

# Define CSV file name for logging results
save_path = "./results/compare"
if not os.path.exists(save_path):
  os.makedirs(save_path)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_filename = f"{save_path}/benchmark_results_{timestamp}.csv"

# Define CSV header
header = ["model_name", "runtime", "tokens_per_second", "overall_score"] + [f"task_{task.name}" for task in tasks]
    
# Open the results CSV file in append mode and write the header if the file is empty
with open(results_filename, mode="a", newline="", encoding = "utf-8") as results_file:
  writer = csv.writer(results_file)
  
  # Run the benchmark for each model, saving results after each benchmark
  for model_name in model_names:
      print(f"Running benchmark for model: {model_name}")
      try:
        result = run_benchmark(model_name, tasks)
        
        if result:
          # Get task scores that were actually run
          task_scores = result["task_scores"]
          
          # Write header if the file is empty
          if results_file.tell() == 0:
            writer.writerow(header)

          # Prepare the row with the task scores
          row = [
            result["model_name"],
            result["runtime"],
            result["tokens_per_second"],
            result["overall_score"],
          ] + [task_scores.get(task.name, "N/A") for task in tasks]
          
          # Write the row to the CSV file and flush
          writer.writerow(row)
          results_file.flush()
      except Exception as ex:
        # On error, write blank row and continue test
        writer.writerow([model_name, "", "", ""] + ["" for task in tasks])
        results_file.flush()


print(f"Benchmark completed. Results saved to '{results_filename}'.")
