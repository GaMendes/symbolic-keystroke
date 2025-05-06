import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import os
import time
import sys
from evaluator import EGPPEvaluator
from typing import List, Dict, Any, Optional
from utils import default_print
from experiment_configuration_model import ExperimentConfigurationModel

class SessionExperimentRunner:

    """
    Orchestrates the evaluation of session-specific eggp models. Loads configuration, iterates through users/sessions, loads data,
    loads models (expressions), evaluates using EGPPEvaluator, calculates metrics, and saves results.
    """

    def __init__(self, config:ExperimentConfigurationModel=None, logger=None):

        self.logger   = logger
        self._info    = self.logger.info if self.logger else default_print
        self._error   = self.logger.error if self.logger else default_print
        self._warning = self.logger.warning if self.logger else default_print

        self.config = config

        if not self.config:
            logger.error(f"Error loading configurations variables!", component="RUN_EVALUATION")
            sys.exit(1)

    def _load_eggp_expression(self, user: str, session: int) -> Optional[str]:

        """
        Loads the eggp expression string from its output CSV file. Assumes filename convention user_<i>_session_<j>_eggp_output.csv
        Assumes expression is in the 'Expression' column of the first data row.
        """

        eggp_output_dir = self.config.eggp_output_dir
        filename  = f"user_{user}_session_{session}_eggp_output.csv"
        file_path = os.path.join(eggp_output_dir, filename)

        if not os.path.exists(file_path):
            self._error(f"    Error: eggp output file not found: {file_path}", component="RUN_EVALUATION")
            return None

        try:
            df_out = pd.read_csv(file_path)

            if 'Expression' not in df_out.columns:
                self._error(f"    Error: 'Expression' column not found in {file_path}", component="RUN_EVALUATION")
                return None

            if len(df_out) == 0:
                 self._error(f"    Error: eggp output file is empty: {file_path}", component="RUN_EVALUATION")
                 return None
    
            # Assuming the first row contains the relevant expression
            expression = df_out['Expression'].iloc[0]
            return str(expression)

        except Exception as e:

            self._error(f"    Error reading or parsing eggp output file {file_path}: {e}", component="RUN_EVALUATION")
            return None

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:

        """Calculates Balanced Accuracy, FMR, FNMR."""

        metrics = {}
        try:
            metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

            # Calculate FMR/FNMR from confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

            # FMR = FP / (FP + TN) = FP / Condition Negative
            fmr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            metrics['fmr'] = fmr

            # FNMR = FN / (FN + TP) = FN / Condition Positive
            fnmr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            metrics['fnmr'] = fnmr

        except Exception as e:

            self._error(f"    Error calculating metrics: {e}", component="RUN_EVALUATION")
            metrics['balanced_accuracy'] = np.nan
            metrics['fmr'] = np.nan
            metrics['fnmr'] = np.nan

        return metrics

    def run_experiments(self):

        """Runs the full evaluation loop."""

        start_run_time = time.time()
        self._info("Starting experiment run...", component="RUN_EVALUATION")

        data_dir    = self.config.preprocessed_data_dir
        target_col  = self.config.target_column_name
        min_session = self.config.min_session
        max_session = self.config.max_session
        threshold_strategy = self.config.threshold_strategy
        default_threshold  = self.config.default_threshold

        try:
            all_files = os.listdir(data_dir)
            # Extract unique user IDs (e.g., 's002' from 'user_s002_session_1_train.csv')
            users = sorted(list(set([f.split('_')[1] for f in all_files if f.startswith('user_') and '_train.csv' in f])))

            if not users:
                 self._error(f"Error: Could not find any preprocessed data files in {data_dir}", component="RUN_EVALUATION")
                 return

            self._info(f"Found {len(users)} users in data directory.", component="RUN_EVALUATION")

        except FileNotFoundError:

            self._error(f"Error: Preprocessed data directory not found: {data_dir}", component="RUN_EVALUATION")
            return

        except Exception as e:

            self._error(f"Error listing users from data directory {data_dir}: {e}", component="RUN_EVALUATION")
            return

        all_results = []

        for user in users:
            self._info(f"\nProcessing User: {user}", component="RUN_EVALUATION")

            for session_j in range(min_session, max_session + 1):
                self._info(f"  Evaluating Model Trained on Session: {session_j}", component="RUN_EVALUATION")

                # 1. Load Test Data
                test_filename = f"user_{user}_session_{session_j}_test.csv"
                test_filepath = os.path.join(data_dir, test_filename)

                if not os.path.exists(test_filepath):
                    self._warning(f"    Warning: Test file not found, skipping: {test_filepath}", component="RUN_EVALUATION")
                    continue

                try:
                    df_test = pd.read_csv(test_filepath)
                    y_true = df_test[target_col].values
                    # Assume features are all columns except the label column
                    feature_cols_test = [col for col in df_test.columns if col != target_col]
                    X_test = df_test[feature_cols_test]

                except Exception as e:
                    self._error(f"    Error loading or processing test file {test_filepath}: {e}", component="RUN_EVALUATION")
                    continue

                # 2. Load eggp Expression
                expression = self._load_eggp_expression(user, session_j)
                if expression is None:
                    self._warning(f"    Skipping evaluation due to missing/invalid eggp output.", component="RUN_EVALUATION")
                    continue

                # 3. Instantiate Evaluator
                try:
                    evaluator = EGPPEvaluator(expression_string=expression, logger=self.logger)
                    # Check if parsing failed
                    if evaluator._lambdified_fx is None:
                         self._warning(f"    Skipping evaluation due to expression parsing error.", component="RUN_EVALUATION")
                         continue
    
                except Exception as e:
                    self._error(f"    Error instantiating EGPPEvaluator: {e}", component="RUN_EVALUATION")
                    continue

                # 4. Prepare Threshold Config (and data for adaptive if needed)
                threshold_config = {
                    'strategy': threshold_strategy,
                    'value': default_threshold
                }

                train_genuine_scores = None # Placeholder for adaptive scores

                # Placeholder logic if adaptive strategy is selected
                # This part needs refinement based on exact adaptive needs
                if threshold_strategy != 'fixed':
                     # TODO: Load TRAINING data (user_<i>_session_<j>_train.csv)
                     # TODO: Filter for genuine samples (label == 1)
                     # TODO: Calculate f(x) scores for these samples using evaluator.evaluate_fx()
                     # TODO: Store these scores in train_genuine_scores
                     self._warning(f"    Warning: Adaptive threshold strategy '{threshold_strategy}' selected but requires loading training data and calculating scores - using placeholder logic.", component="RUN_EVALUATION")
                     # train_genuine_scores = ... # Load and calculate here
                     pass # Pass None for now

                # 5. Get Predictions
                try:
                    y_pred = evaluator.predict(X_test, threshold_config, train_genuine_scores)

                    if y_pred is None:
                        self._info("    Skipping evaluation due to prediction error.", component="RUN_EVALUATION")
                        continue

                except Exception as e:
                    self._error(f"    Error during prediction: {e}", component="RUN_EVALUATION")
                    continue

                # 6. Calculate Metrics
                metrics = self._calculate_metrics(y_true, y_pred)

                # 7. Store Results
                result_row = {
                    'user': user,
                    'train_session': session_j,
                    'threshold_strategy': threshold_strategy,
                    'threshold_value': threshold_config.get('value'), # Store used threshold
                    'balanced_accuracy': metrics.get('balanced_accuracy'),
                    'fmr': metrics.get('fmr'),
                    'fnmr': metrics.get('fnmr')
                }
                all_results.append(result_row)
                self._info(f"    Metrics: BAcc={metrics.get('balanced_accuracy'):.4f}, FMR={metrics.get('fmr'):.4f}, FNMR={metrics.get('fnmr'):.4f}", component="RUN_EVALUATION")


        # 8. Save Aggregate Results
        self.save_results(all_results)

        end_run_time = time.time()
        self._info(f"\nExperiment run finished in {end_run_time - start_run_time:.2f} seconds.", component="RUN_EVALUATION")

    def save_results(self, results_list: List[Dict[str, Any]]):

        """Saves the collected results to a CSV file."""

        if not results_list:
            self._warning("No results were generated to save.", component="RUN_EVALUATION")
            return

        results_dir = self.config.evaluation_results_dir
        results_filename = self.config.evaluation_results_filename
        output_path = os.path.join(results_dir, results_filename)

        # Create directory if it doesn't exist
        if not os.path.exists(results_dir):
            self._info(f"Creating results directory: {results_dir}", component="RUN_EVALUATION")
            os.makedirs(results_dir)

        self._info(f"\nSaving evaluation results to {output_path}...", component="RUN_EVALUATION")

        try:
            df_results = pd.DataFrame(results_list)
            df_results.to_csv(output_path, index=False, header=True)
            self._info("Results saved successfully.", component="RUN_EVALUATION")

        except Exception as e:
            self._error(f"Error saving results file {output_path}: {e}", component="RUN_EVALUATION")

if __name__ == "__main__":
    runner = SessionExperimentRunner(config_path='config.txt')
    runner.run_experiments()