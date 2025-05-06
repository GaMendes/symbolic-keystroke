import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import configparser
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ResultVisualizer:

    """
    Analyzes and visualizes experiment results from the metrics CSV file.
    Generates summary statistics and per-user performance plots.
    """

    def __init__(self, run_directory_path: str, config_path: str = 'config.txt'):

        """
        Initializes the visualizer by loading configuration and results data.

        Args:
            run_directory_path: Path to the specific experiment run directory.
            config_path: Path to the configuration file to find results filename.
        """

        self.run_directory = run_directory_path
        self.config = self._load_config(config_path)
        self.results_df: Optional[pd.DataFrame] = None
        self.plots_dir: Optional[str] = None

        if not os.path.isdir(self.run_directory):
            logging.error(f"Experiment run directory not found: {self.run_directory}")
            raise FileNotFoundError(f"Directory not found: {self.run_directory}")

        self._load_results_data()
        self._setup_plots_dir()

    def _load_config(self, config_path: str) -> Optional[configparser.ConfigParser]:

        """Loads configuration to find results filename."""

        if not os.path.exists(config_path):
            logging.warning(f"Config file not found at {config_path}. Using default results filename.")
            return None
        try:
            config = configparser.ConfigParser()
            config.read(config_path)
            return config
        except configparser.Error as e:
            logging.warning(f"Error reading config file {config_path}: {e}. Using default results filename.")
            return None

    def _load_results_data(self):

        """Loads the aggregated metrics CSV file."""

        results_filename = "session_evaluation_metrics.csv" 
        results_subdir = "results"

        if self.config:
            try:

              results_subdir = self.config.get('Paths', 'EVALUATION_RESULTS_SUBDIR', fallback=results_subdir)
              results_filename = self.config.get('Paths', 'EVALUATION_RESULTS_FILENAME', fallback=results_filename)

            except (configparser.NoSectionError, configparser.NoOptionError):
                 logging.warning("Could not read result path/filename from config, using defaults.")

        results_csv_path = os.path.join(self.run_directory, results_subdir, results_filename)

        logging.info(f"Attempting to load results from: {results_csv_path}")
        if not os.path.exists(results_csv_path):
            logging.error(f"Results CSV file not found: {results_csv_path}")
            self.results_df = None
            return

        try:
            self.results_df = pd.read_csv(results_csv_path)
            logging.info(f"Successfully loaded results data. Shape: {self.results_df.shape}")

            required_cols = ['user', 'train_session', 'balanced_accuracy', 'fmr', 'fnmr']
            if not all(col in self.results_df.columns for col in required_cols):
                logging.error(f"Results CSV missing one or more required columns: {required_cols}")
                self.results_df = None
        except Exception as e:
            logging.error(f"Error reading results CSV file {results_csv_path}: {e}", exc_info=True)
            self.results_df = None

    def _setup_plots_dir(self):
        """Creates the plots subdirectory within the run directory."""
        if self.results_df is None: return
        self.plots_dir = os.path.join(self.run_directory, "plots")
        try:
            os.makedirs(self.plots_dir, exist_ok=True)
            logging.info(f"Plots will be saved in: {self.plots_dir}")
        except OSError as e:
            logging.error(f"Could not create plots directory {self.plots_dir}: {e}")
            self.plots_dir = None

    def print_summary_table(self):
        """Calculates and prints average metrics per session."""
        if self.results_df is None or self.results_df.empty:
            logging.warning("No results data loaded, cannot print summary table.")
            return

        logging.info("\n--- Average Performance Metrics per Training Session ---")
        try:
            summary = self.results_df.groupby('train_session')[
                ['balanced_accuracy', 'fmr', 'fnmr']
            ].mean()
            # Format for better readability
            summary_formatted = summary.applymap(lambda x: f"{x:.4f}")
            print(summary_formatted.to_string())
        except KeyError:
            logging.error("Could not calculate summary: Missing required metric columns.")
        except Exception as e:
            logging.error(f"Error calculating summary table: {e}", exc_info=True)

    def print_performance_change_summary(self):
        """Identifies users with largest increase/decrease in Balanced Accuracy."""
        if self.results_df is None or self.results_df.empty:
            logging.warning("No results data loaded, cannot analyze performance change.")
            return

        logging.info("\n--- Balanced Accuracy Change (Session 1 vs. Session 7) ---")
        try:
            # Ensure 'train_session' is numeric if needed
            self.results_df['train_session'] = pd.to_numeric(self.results_df['train_session'], errors='coerce')
            min_sess = self.results_df['train_session'].min()
            max_sess = self.results_df['train_session'].max()

            if pd.isna(min_sess) or pd.isna(max_sess) or min_sess == max_sess:
                 logging.warning(f"Cannot calculate change: Insufficient session range found ({min_sess} to {max_sess}).")
                 return

            logging.info(f"Comparing performance between session {int(min_sess)} and {int(max_sess)}.")

            # Pivot table to get BAcc per user per session easily
            perf_pivot = self.results_df.pivot_table(
                index='user', columns='train_session', values='balanced_accuracy'
            )

            # Calculate change only for users present in both min and max session
            valid_users_perf = perf_pivot[[min_sess, max_sess]].dropna()
            if valid_users_perf.empty:
                 logging.warning("No users found with data for both first and last sessions.")
                 return

            valid_users_perf['bacc_change'] = valid_users_perf[max_sess] - valid_users_perf[min_sess]

            # Find max increase and decrease
            max_increase_user = valid_users_perf['bacc_change'].idxmax()
            max_increase_val = valid_users_perf['bacc_change'].max()
            max_decrease_user = valid_users_perf['bacc_change'].idxmin()
            max_decrease_val = valid_users_perf['bacc_change'].min()

            print(f"Largest Increase: User {max_increase_user} ({max_increase_val:+.4f})")
            print(f"Largest Decrease: User {max_decrease_user} ({max_decrease_val:+.4f})")

        except KeyError:
            logging.error("Could not analyze change: Missing 'user', 'train_session', or 'balanced_accuracy' columns.")
        except Exception as e:
            logging.error(f"Error analyzing performance change: {e}", exc_info=True)


    def generate_user_plots(self):
        """Generates and saves a performance trend plot for each user."""
        if self.results_df is None or self.results_df.empty:
            logging.warning("No results data loaded, cannot generate plots.")
            return
        if not self.plots_dir:
             logging.warning("Plots directory not available, skipping plot generation.")
             return

        logging.info("\n--- Generating Per-User Performance Plots ---")
        unique_users = self.results_df['user'].unique()

        for i, user_id in enumerate(unique_users):
            if (i + 1) % 10 == 0 or i == 0 or len(unique_users) < 10: # Log progress
                 logging.info(f"  Generating plot for user {i+1}/{len(unique_users)}: {user_id}")

            user_data = self.results_df[self.results_df['user'] == user_id].sort_values('train_session')

            if user_data.empty:
                logging.warning(f"  No data found for user {user_id}, skipping plot.")
                continue

            plt.figure(figsize=(10, 6))
            plt.plot(user_data['train_session'], user_data['balanced_accuracy'], marker='o', linestyle='-', label='Balanced Accuracy')
            plt.plot(user_data['train_session'], user_data['fmr'], marker='s', linestyle='--', label='FMR')
            plt.plot(user_data['train_session'], user_data['fnmr'], marker='^', linestyle=':', label='FNMR')

            plt.title(f"User {user_id}: Performance Metrics vs. Training Session")
            plt.xlabel("Training Session")
            plt.ylabel("Metric Value")
            plt.xticks(user_data['train_session'].unique())
            plt.ylim(0, 1.05) # Y-axis from 0 to 1 (slightly above for legend)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.legend()

            # Save the plot
            plot_filename = f"user_{user_id}_metrics.png"
            plot_filepath = os.path.join(self.plots_dir, plot_filename)
            try:
                plt.savefig(plot_filepath)
            except Exception as e:
                logging.error(f"  Failed to save plot {plot_filepath}: {e}")
            plt.close() # Close the figure to free memory

        logging.info("Finished generating plots.")


    def run_analysis(self):
        """Runs all analysis and visualization steps."""
        if self.results_df is not None:
            self.print_summary_table()
            self.print_performance_change_summary()
            self.generate_user_plots()
        else:
            logging.error("Analysis cannot proceed as results data failed to load.")

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize EGGp Keystroke Experiment Results.")
    parser.add_argument("run_directory", type=str,
                        help="Path to the specific experiment run directory containing the results CSV.")
    parser.add_argument("--config", type=str, default='config.txt',
                        help="Path to the configuration file (used to find results filename).")

    args = parser.parse_args()

    try:
        visualizer = ResultVisualizer(run_directory_path=args.run_directory, config_path=args.config)
        visualizer.run_analysis()
    except FileNotFoundError:
        # Error already logged by constructor
        pass
    except Exception as e:
        logging.error(f"An unexpected error occurred during visualization: {e}", exc_info=True)

