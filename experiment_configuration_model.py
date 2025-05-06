from dataclasses import dataclass

@dataclass
class ExperimentConfigurationModel:
  """
  Data structure to hold processed configuration values for the experiment workflow.
  """
  # Paths
  input_csv_path: str             # Path to the raw input CSV (e.g., DSL-StrongPasswordData.csv)
  preprocessed_data_dir: str      # Directory for intermediate train/test CSVs
  eggp_output_dir: str            # Directory where eggp saves its output CSVs (with Expression)
  evaluation_results_dir: str     # Directory to save the final evaluation metrics CSV

  # Experiment Parameters
  min_session: int                # First session index to process for training
  max_session: int                # Last session index to process for training
  target_column_name: str         # Name of the target/label column in CSVs

  # Evaluation Parameters (needed by SessionExperimentRunner)
  evaluation_results_filename: str # Filename for the final metrics CSV
  threshold_strategy: str          # e.g., 'fixed', 'adaptive_placeholder_min', etc.
  default_threshold: float         # The threshold value used for the 'fixed' strategy