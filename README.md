# Keystroke Dynamics EGGp Experiment Workflow

This project implements a workflow for training and evaluating user authentication models based on keystroke dynamics using the `eggp` tool. We use the CMU Keystroke Dynamics dataset.

## Project Goal

The primary goal is to train a separate binary classifier for each user for different training sessions (e.g., session 1, session 2, ..., session 7) and evaluate its performance on data from the immediately following session (e.g., train on session `j`, test on session `j + 1`). This allows for the analysis of how well models generalize to future user behavior.

## Features

* **Session-Specific Preprocessing:** Splits the raw dataset into training and testing files for each user and session. Training sets are balanced (50 genuine vs 50 impostor samples), while test sets use all available data from the subsequent session.
* **EGGp Integration:** Automates the execution of the external `./eggp` binary for each user/session pair.
* **Model Evaluation:** Parses the symbolic expression output from `eggp`, applies a sigmoid function, and evaluates classification performance using Balanced Accuracy, False Match Rate (FMR), and False Non-Match Rate (FNMR).
* **Result Visualization:** Generates per-user plots showing performance metrics across training sessions, average trends, and metric distributions.
* **Configuration File:** Uses `config.txt` for setting up paths, parameters, and `eggp` arguments.
* **Structured Logging:** Logs workflow progress and errors to both the console and a detailed `experiment_data.csv` file within the run directory.

## Dependencies

1.  **Python:** Python 3.7+ recommended.
2.  **Python Libraries:** Install using pip:
    ```bash
    pip install -r requirements.txt
    ```
    (Requires `pandas`, `numpy`, `scikit-learn`, `sympy`, `matplotlib`, `seaborn`)
3.  **EGGp Binary:**
    * The compiled `eggp-2.0.1.2-Linux-ghc-9.10.1` executable ([https://github.com/folivetti/srtree/tree/main/apps/eggp](https://github.com/folivetti/srtree/tree/main/apps/eggp)) must be present in the project's bin directory and named `eggp`.
    * Ensure `eggp`'s own dependencies (like `libgmp`, `libnlopt`, `zlib`, and potentially Haskell build tools if compiling from source) are installed on your system. Refer to the `eggp` documentation for its specific installation requirements.
4.  **Dataset:** The raw `DSL-StrongPasswordData.csv` file from the CMU Keystroke Dynamics dataset. By default, the workflow expects it at `data/DSL-StrongPasswordData.csv` (configurable in `config.txt`). The dataset is available at ([https://www.cs.cmu.edu/~keystroke/](https://www.cs.cmu.edu/~keystroke/)).

## Project Structure

``` bash
.
├── config.txt                  # Configuration file
├── requirements.txt            # Python dependencies
├── main_workflow.py            # Main script to orchestrate the workflow
├── data_processor.py           # Class for preprocessing data
├── evaluator.py                # Class for evaluating EGGp expressions
├── run_evaluation.py           # Class for running the evaluation phase (contains ExperimentRunConfig dataclass)
├── visualize_results.py        # Class for generating result summaries and plots
├── experiment_logger.py        # Class for CSV 
├── bin
│   └── eggp                    # The compiled EGGp binary (place here)
└── results                    # Base directory for experiment run outputs
│   └── experiment_YYYYMMDD_HHMMSS/    # Unique directory for each run
│       └── data                # Output of data_processor.py
│       └── eggp_output         # Output of EGGp runs
│       └── config.txt          # configuration from the experiment
│       └── experiment_data.csv # Detailed CSV log for this run (from CsvExperimentLogger)
│       └── session_evaluation_metrics.csv
├── data/
│   └── DSL-StrongPasswordData.csv # Raw dataset (place here or configure path)
```

## Configuration (`config.txt`)

This file controls various aspects of the workflow:

* **`[Paths]`**: Defines the base results directory, subdirectory names for intermediate/final outputs, filenames for logs/results, and the path to the raw input CSV.
* **`[Evaluation]`**: Sets the threshold strategy (e.g., `fixed`), the default threshold value, and the name of the target column in the data files.
* **`[Experiment]`**: Defines the range of training sessions to process (`MIN_SESSION`, `MAX_SESSION`) and potentially parameters like `N_TRAIN_SAMPLES_PER_CLASS` and `RANDOM_STATE`.
* **`[EGGpArgs]`**: Specifies the command-line arguments (like `maxSize`, `nPop`, `pc`, `pm`, `generations`) to be passed to the `./eggp` executable.

## How to Run

Execute the main workflow script from your terminal in the project's root directory.

**1. Full Workflow (New Run):**

```bash
python main_workflow.py
```
