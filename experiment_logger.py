import pandas as pd
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

class ExperimentLogger:

    """
    Handles standardized logging for experiments. Logs messages to the console immediately and aggregates log records
    to be saved to a single CSV file at the end of the experiment.
    """

    def __init__(self, csv_log_path: str):

        """
        Initializes the logger.

        Args:
            csv_log_path: The full path where the final log CSV should be saved.
        """
  
        self.csv_log_path = csv_log_path
        self.log_records: List[Dict[str, Any]] = []
        self._log_columns = ['timestamp', 'level', 'component', 'user_id', 'session_id', 'message']
  
        print(f"Logger initialized. Log entries will be saved to '{self.csv_log_path}' upon calling save_log().")

    def _log(self, level: str, message: str, component: Optional[str],
              user_id: Optional[str], session_id: Optional[int]):

        """Internal method to process and store log entries."""

        timestamp = datetime.now()
        timestamp_str = timestamp.isoformat(sep=' ', timespec='milliseconds')
        log_prefix = f"[{timestamp_str}] [{level:<7}]"

        if component:
            log_prefix += f" [{component}]"

        context = []
        if user_id:
            context.append(f"User: {user_id}")

        if session_id is not None:
            context.append(f"Session: {session_id}")

        if context:
            log_prefix += f" [{' | '.join(context)}]"

        print(f"{log_prefix} {message}")

        log_entry = {
            'timestamp': timestamp_str,
            'level': level,
            'component': component if component else 'GENERAL', # Default component
            'user_id': user_id,
            'session_id': session_id,
            'message': message
        }
        self.log_records.append(log_entry)


    def info(self, message: str, component: str = 'GENERAL', user_id: str = None, session_id: int = None):
        """Logs an informational message."""
        self._log('INFO', message, component, user_id, session_id)

    def warning(self, message: str, component: str = 'GENERAL', user_id: str = None, session_id: int = None):
        """Logs a warning message."""
        self._log('WARNING', message, component, user_id, session_id)

    def error(self, message: str, component: str = 'GENERAL', user_id: str = None, session_id: int = None):
        """Logs an error message."""
        self._log('ERROR', message, component, user_id, session_id)

    def save_log(self):

        """
        Saves all accumulated log records to the specified CSV file.
        Should be called once at the end of the experiment workflow.
        """

        if not self.log_records:
            print("No log records generated to save.")
            return

        print(f"\nSaving {len(self.log_records)} log entries to {self.csv_log_path}...")
        try:

            log_dir = os.path.dirname(self.csv_log_path)
            if log_dir and not os.path.exists(log_dir):
                print(f"Creating log directory: {log_dir}")
                os.makedirs(log_dir)

            df_log = pd.DataFrame(self.log_records, columns=self._log_columns)

            df_log.to_csv(self.csv_log_path, index=False, header=True)
            print(f"Log successfully saved to {self.csv_log_path}")

        except Exception as e:
            print(f"Error saving log file {self.csv_log_path}: {e}")