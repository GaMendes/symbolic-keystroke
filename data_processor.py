import pandas as pd
import os
import time
from typing import List, Optional
from utils import default_print

class SessionDataPreprocessor:

    """
    Preprocesses the raw keystroke data to generate session-specific, binary-labeled datasets.
    Training sets are balanced (N genuine vs N impostor). Test sets use ALL genuine vs ALL impostors from the subsequent session.
    Outputs CSV files with headers.
    """

    def __init__(self, input_csv_path: str,
                 output_dir: str,
                 n_train_samples_per_class: int = 50,
                 sessions_to_train: range = range(1, 8),
                 random_state: int = 42,
                 target_col_name: str = "class",
                 logger=None):

        self.input_csv_path = input_csv_path
        self.output_dir = output_dir
        self.n_train_samples_per_class = n_train_samples_per_class
        self.sessions_to_train = sessions_to_train
        self.random_state = random_state
        self.feature_columns: List[str] = []
        self.subject_column = 'subject'
        self.session_column = 'sessionIndex'
        self.rep_column = 'rep'
        self.target_col_name = target_col_name

        self.logger   = logger
        self._info    = self.logger.info if self.logger else default_print
        self._error   = self.logger.error if self.logger else default_print
        self._warning = self.logger.warning if self.logger else default_print

    def _load_data(self) -> Optional[pd.DataFrame]:
        """Loads and sorts the initial dataset."""

        self._info(f"Loading data from {self.input_csv_path}...", component="PREPROCESSING")
        try:
            df_full = pd.read_csv(self.input_csv_path)
            self._info(f"Data loaded successfully. Shape: {df_full.shape}", component="PREPROCESSING")

            required_cols = [self.subject_column, self.session_column, self.rep_column]
            try:

                start_feature_col = df_full.columns.get_loc('H.period')
                end_feature_col = df_full.columns.get_loc('H.Return')
                self.feature_columns = df_full.columns[start_feature_col : end_feature_col + 1].tolist()

                self._info(f"Identified {len(self.feature_columns)} feature columns.", component="PREPROCESSING")

            except KeyError as e:
                self._error(f"Error: Column {e} not found. Check CSV header.", component="PREPROCESSING")
                return None

            self._info(f"Sorting data by {required_cols}...", component="PREPROCESSING")
            df_full.sort_values(by=required_cols, inplace=True)
            self._info("Sorting complete.", component="PREPROCESSING")

            return df_full

        except FileNotFoundError:

            self._error(f"Error: Input file '{self.input_csv_path}' not found.", component="PREPROCESSING")
            return None

        except Exception as e:

            self._error(f"An error occurred during data loading: {e}", component="PREPROCESSING")
            return None

    def _create_child_csv(self, df: pd.DataFrame, filename: str):
        """Saves a DataFrame to CSV with header, excluding index."""

        self._info(f"    Saving {len(df)} instances to {filename}...", component="PREPROCESSING")
        try:
            if self.target_col_name in df.columns:
                 df[self.target_col_name] = df[self.target_col_name].astype(int)

            df.to_csv(filename, index=False, header=True)

        except Exception as e:
            self._info(f"    Error saving file {filename}: {e}", component="PREPROCESSING")

    def process_and_save(self):
        """Runs the entire preprocessing pipeline."""

        start_time = time.time()
        df_full = self._load_data()

        if df_full is None:
            self._error("Preprocessing aborted due to loading errors.", component="PREPROCESSING")
            return

        if not os.path.exists(self.output_dir):
            self._info(f"Creating output directory: {self.output_dir}", component="PREPROCESSING")
            os.makedirs(self.output_dir)

        unique_users = df_full[self.subject_column].unique()
        n_users = len(unique_users)
        self._info(f"Found {n_users} unique users.", component="PREPROCESSING")

        files_created = 0
        for user_idx, target_user in enumerate(unique_users):
            self._info(f"\nProcessing User {user_idx+1}/{n_users}: {target_user}...", component="PREPROCESSING")

            for session_j in self.sessions_to_train:
                session_k = session_j + 1 # Test session
                self._info(f"  Processing Training Session {session_j} (Testing on Session {session_k})...", component="PREPROCESSING")

                # Filter data for sessions j and k
                df_session_j = df_full[df_full[self.session_column] == session_j]
                df_session_k = df_full[df_full[self.session_column] == session_k]

                # Get genuine/impostor pools for session j (Training)
                df_genuine_pool_j  = df_session_j[df_session_j[self.subject_column] == target_user]
                df_impostor_pool_j = df_session_j[df_session_j[self.subject_column] != target_user]

                # Get genuine/impostor pools for session k (Testing)
                df_genuine_pool_k  = df_session_k[df_session_k[self.subject_column] == target_user]
                df_impostor_pool_k = df_session_k[df_session_k[self.subject_column] != target_user]

                # Need enough samples for the balanced training set
                if len(df_genuine_pool_j) < self.n_train_samples_per_class:
                    self._warning(f"    Skipping: Insufficient genuine samples ({len(df_genuine_pool_j)}) for user {target_user} in training session {session_j}.", component="PREPROCESSING")
                    continue

                if len(df_impostor_pool_j) < self.n_train_samples_per_class:
                    self._warning(f"    Skipping: Insufficient impostor samples ({len(df_impostor_pool_j)}) in pool for training session {session_j}.", component="PREPROCESSING")
                    continue

                # Need *some* data for testing session k
                if df_genuine_pool_k.empty or df_impostor_pool_k.empty:
                    self._warning(f"    Skipping: No genuine or no impostor samples found for user {target_user} in testing session {session_k}.", component="PREPROCESSING")
                    continue

                columns_to_save = [self.target_col_name] + self.feature_columns
                try:
                    train_genuine_pool  = df_genuine_pool_j.head(self.n_train_samples_per_class).copy()
                    train_genuine_pool[self.target_col_name] = 1

                    train_impostor_pool = df_impostor_pool_j.sample(n=self.n_train_samples_per_class, random_state=self.random_state).copy()
                    train_impostor_pool[self.target_col_name] = 0

                    df_train = pd.concat([train_genuine_pool, train_impostor_pool])

                    #remove unecessary columns(rep, subject, etc.) and shuffle data inside the dataset
                    df_train = df_train[columns_to_save].sample(frac=1, random_state=self.random_state).reset_index(drop=True)

                    train_filename = os.path.join(self.output_dir, f"user_{target_user}_session_{session_j}_train.csv")
                    self._create_child_csv(df_train, train_filename)
                    files_created += 1

                except Exception as e:
                    self._error(f"    ERROR generating training set for user {target_user}, session {session_j}: {e}", component="PREPROCESSING")
                    continue

                try:
                    test_genuine_pool = df_genuine_pool_k.copy()
                    test_genuine_pool[self.target_col_name] = 1

                    test_impostor_pool = df_impostor_pool_k.copy()
                    test_impostor_pool[self.target_col_name] = 0

                    df_test = pd.concat([test_genuine_pool, test_impostor_pool])
                    df_test = df_test[columns_to_save].sample(frac=1, random_state=self.random_state).reset_index(drop=True)


                    test_filename = os.path.join(self.output_dir, f"user_{target_user}_session_{session_j}_test.csv")
                    self._create_child_csv(df_test, test_filename)
                    files_created += 1

                except Exception as e:
                     self._error(f"    ERROR generating testing set for user {target_user}, session {session_j} (using ALL data from session {session_k}): {e}", component="PREPROCESSING")
                     continue

        end_time = time.time()
        self._info(f"\nPreprocessing finished in {end_time - start_time:.2f} seconds.", component="PREPROCESSING")
        self._info(f"Total files created: {files_created}", component="PREPROCESSING")
        self._info(f"Output files generated in directory: {self.output_dir}", component="PREPROCESSING")
