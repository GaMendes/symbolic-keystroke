import configparser
import os
import subprocess
import time
import sys
from experiment_logger import ExperimentLogger
from experiment_configuration_model import ExperimentConfigurationModel

try:
    from data_processor import SessionDataPreprocessor
    from run_evaluation import SessionExperimentRunner
except ImportError as e:
    print(f"Error importing necessary classes: {e}")
    print("Ensure data_processor.py and run_evaluation.py are in the same directory.")
    sys.exit(1)


CONFIG_FILE = "config.txt"
EGGP_EXECUTABLE = "./bin/eggp"

def load_config(config_path):

    """Loads configuration from the specified file."""

    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return None

    try:
        config = configparser.ConfigParser()
        config.read(config_path)

        required_sections = ["Paths", "Evaluation", "Experiment"]
        if not all(section in config for section in required_sections):
            print(f"Error: Config file {config_path} missing required sections.")
            return None

        return config

    except configparser.Error as e:
        print(f"Error reading config file {config_path}: {e}")
        return None

def get_user_list(data_dir):

    """Gets the list of unique user IDs from preprocessed filenames."""

    users = set()
    try:
        for filename in os.listdir(data_dir):
            if filename.startswith("user_") and "_train.csv" in filename:
                parts = filename.split("_")
                if len(parts) >= 2:
                    users.add(parts[1])

        if not users:
            print(f"Warning: No user files found in {data_dir} to determine user list.")
            return []

        return sorted(list(users))

    except FileNotFoundError:
        print(f"Error: Preprocessed data directory not found: {data_dir}")
        return []

    except Exception as e:
        print(f"Error getting user list from {data_dir}: {e}")
        return []

def get_config_value(config: configparser.ConfigParser,
                     section: str,
                     option: str,
                     fallback: str,
                     logger: ExperimentLogger) -> str:
    """
    Gets a value from config, logging whether the file value or fallback was used.

    Args:
        config: The ConfigParser object.
        section: The section name.
        option: The option name.
        fallback: The fallback value to use if the option is not found.

    Returns:
        The retrieved value (either from file or fallback).
    """

    if (config.has_section(section) and config.has_option(section, option)):
        value = config.get(section, option)
        logger.info(f"Config: Using '{option}' from section '[{section}]': '{value}'")
        return value
    else:
        logger.warning(f"Config: Option '{option}' not found in section '[{section}]'. Using fallback: '{fallback}'")
        return fallback

def main():

    """Executes the entire experiment workflow."""

    overall_start_time = time.time()

    # 1. Load Configuration
    print(f"\n--- Loading Configuration ({CONFIG_FILE}) ---")
    config = load_config(CONFIG_FILE)

    log_file_path = os.path.join(config.get("Paths", "EVALUATION_RESULTS_DIR", fallback="results"),
                             "experiment_data.csv")
    logger = ExperimentLogger(csv_log_path=log_file_path)

    if config is None:
        sys.exit(1)

    logger.info("Configuration loaded.")
    logger.info("Starting Main Workflow...")

    try:
        input_csv        = get_config_value(config, "Paths", "INPUT_DATA_FILE", "data/DSL-StrongPasswordData.csv", logger)
        preprocessed_dir = get_config_value(config, "Paths", "PREPROCESSED_DATA_DIR", "results/data", logger)
        eggp_output_dir  = get_config_value(config, "Paths", "EGGP_OUTPUT_DIR", "results/eggp_output", logger)
        evaluations_dir  = get_config_value(config, "Paths", "EVALUATION_RESULTS_DIR", "results", logger)
        results_filename = get_config_value(config, "Paths", "EVALUATION_RESULTS_FILENAME", "session_evaluation_metrics.csv", logger)
        min_session      = int(get_config_value(config, "Experiment", "MIN_SESSION", 1, logger))
        max_session      = int(get_config_value(config, "Experiment", "MAX_SESSION", 7, logger))
        target_col_name  = get_config_value(config, "Evaluation", "TARGET_COLUMN_NAME", "class", logger)
        threshold_strategy  = get_config_value(config, "Evaluation", "THRESHOLD_STRATEGY", "fixed", logger)
        default_threshold   = float(get_config_value(config, "Evaluation", "DEFAULT_THRESHOLD", 0.5, logger))

        # eggp command arguments (example - customize as needed!)
        eggp_args = [
            "--maxSize", get_config_value(config, "EGGpArgs", "maxSize", 100, logger),
            "--nPop", get_config_value(config, "EGGpArgs", "nPop", 100, logger),
            "--pc", get_config_value(config, "EGGpArgs", "pc", 0.9, logger),
            "--pm", get_config_value(config, "EGGpArgs", "pm", 0.1, logger),
            "-g", get_config_value(config, "EGGpArgs", "generations", 50, logger)
        ]

        experiment_configuration = ExperimentConfigurationModel(
            input_csv_path=input_csv,
            preprocessed_data_dir=preprocessed_dir,
            eggp_output_dir=eggp_output_dir,
            evaluation_results_dir=evaluations_dir,
            evaluation_results_filename=results_filename,
            min_session=min_session,
            max_session=max_session,
            target_column_name=target_col_name,
            threshold_strategy=threshold_strategy,
            default_threshold=default_threshold
        )

        # Check if eggp binary exists
        if not os.path.isfile(EGGP_EXECUTABLE):
             logger.error(f"Error: eggp executable not found at {EGGP_EXECUTABLE}")
             sys.exit(1)

        os.makedirs(eggp_output_dir, exist_ok=True)
        os.makedirs(evaluations_dir, exist_ok=True)

    except (configparser.NoSectionError, configparser.NoOptionError, ValueError) as e:
        logger.error(f"Error accessing configuration value: {e}")
        sys.exit(1)


    logger.info(f"\n--- Phase 1: Data Preprocessing ---")
    try:
        preprocessor = SessionDataPreprocessor(
            input_csv_path=input_csv,
            output_dir=preprocessed_dir,
            sessions_to_train=range(min_session, max_session + 1),
            target_col_name=target_col_name,
            logger=logger
        )

        preprocessor.process_and_save()

    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        sys.exit(1)

    logger.info("Data preprocessing phase complete.")


    #3. Run eggp Externally
    logger.info(f"\n--- Phase 2: Running eggp ({EGGP_EXECUTABLE}) ---")
    users = get_user_list(preprocessed_dir)

    if not users:
         logger.error("Error: Could not determine user list from preprocessed data. Aborting eggp runs.")
         sys.exit(1)

    eggp_runs_failed = 0

    for user in users:
        for session_j in range(min_session, max_session + 1):
            input_train_file = os.path.join(preprocessed_dir, f"user_{user}_session_{session_j}_train.csv")
            input_test_file  = os.path.join(preprocessed_dir, f"user_{user}_session_{session_j}_test.csv")
            eggp_output_file = os.path.join(eggp_output_dir, f"user_{user}_session_{session_j}_eggp_output.csv")

            if not os.path.exists(input_train_file):
                logger.warning(f"  Warning: Input train file not found, skipping eggp run: {input_train_file}")
                continue

            if not os.path.exists(input_test_file):
                logger.warning(f"  Warning: Input test file not found, skipping eggp run: {input_test_file}")
                continue

            logger.info(f"  Running eggp for User {user}, Session {session_j}...")

            # Assumes eggp uses all other columns as features by default
            dataset_arg = f"{input_train_file}:::{target_col_name}"
            test_arg    = f"{input_test_file}:::{target_col_name}"

            command = [
                EGGP_EXECUTABLE,
                '--dataset', dataset_arg,
                '--test', test_arg,
            ] + eggp_args 

            try:
                result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=300)
                logger.info(f"    eggp completed successfully. Saving output to {eggp_output_file}")

                with open(eggp_output_file, "w") as f_out:
                    f_out.write(result.stdout.strip() + "\n")

            except FileNotFoundError:

                 logger.error(f"    ERROR: eggp executable not found at '{EGGP_EXECUTABLE}'. Check path.")
                 eggp_runs_failed += 1

            except subprocess.CalledProcessError as e:

                logger.error(f"    ERROR: eggp run failed for User {user}, Session {session_j}.")
                logger.error(f"    Command: {' '.join(e.cmd)}")
                logger.error(f"    Return Code: {e.returncode}")
                logger.error(f"    Stderr: {e.stderr.strip()}")
                logger.error(f"    Stdout: {e.stdout.strip()}")

                eggp_runs_failed += 1
    
            except subprocess.TimeoutExpired:

                 logger.error(f"    ERROR: eggp run timed out for User {user}, Session {session_j}.")
                 eggp_runs_failed += 1

            except Exception as e:

                 logger.error(f"    An unexpected error occurred running eggp for User {user}, Session {session_j}: {e}")
                 eggp_runs_failed += 1

    if eggp_runs_failed > 0:
        logger.warning(f"Warning: {eggp_runs_failed} eggp runs failed. Evaluation might be incomplete.")

    logger.info("eggp execution phase complete.")
    logger.info(f"\n--- Phase 3: Evaluating Results ---")

    try:

        eval_runner = SessionExperimentRunner(config=experiment_configuration, logger=logger)
        eval_runner.run_experiments()

    except Exception as e:

        logger.error(f"Error during evaluation phase: {e}")
        sys.exit(1)

    logger.info("Evaluation phase complete.")

    overall_end_time = time.time()
    logger.info(f"\n--- Main Workflow Completed ---")
    logger.info(f"Total execution time: {overall_end_time - overall_start_time:.2f} seconds.")

    logger.save_log()

if __name__ == "__main__":
    main()