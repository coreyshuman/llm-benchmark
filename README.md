# LLM Benchmarking Using Deepeval and Ollama

This project makes it simple to perform local benchmarks on LLMs.
It features a single-model test with detailed results and a multi-model comparison test.

## Requirements

- Python
- Ollama

## Setup

- Run bootstrap to create virtual environment and install dependencies.

```
python bootstrap.py
```

## Run Benchmarks

Model names can be found on [Ollama](https://ollama.com/search).
The Ollama server should be running before executing these tests.
Ollama will automatically download models as necessary.

Testing speed will be relative to the capabilities of your hardware. Recommend an appropriately sized GPU for the model you want to run. Estimated recommendations:

```
8B  = 8GB
14B = 16GB
32B = 32GB
70B = 48GB
```

### Test Individual LLM


```
python benchmark-single.py <model_name> [<task1> <task2> ...]
```

### Compare Multiple LLMs


```
python benchmark-compare.py [<model1> <model2> ...] --tasks [<task1> <task2> ...]
```