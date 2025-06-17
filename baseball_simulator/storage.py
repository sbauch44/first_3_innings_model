# storage.py
import logging
from pathlib import Path
import io # Added for S3 stream handling
import os # For os.path.join for S3 like paths

import boto3 # Added for S3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError # Added for S3 error handling
import polars as pl
import pandas as pd # Added for pandas DataFrame support

import config  # Assuming config.py is in the same directory

# Configure logging for this module
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- S3 Helper Functions ---
def is_s3_path(path: str) -> bool:
    """Checks if the given path is an S3 path."""
    return isinstance(path, str) and path.startswith("s3://")

def parse_s3_path(s3_path: str) -> tuple[str, str]:
    """Parses an S3 path into bucket name and object key."""
    if not is_s3_path(s3_path):
        raise ValueError(f"Invalid S3 path: {s3_path}. Must start with 's3://'")
    path_parts = s3_path.replace("s3://", "").split("/", 1)
    if len(path_parts) < 2: # Ensure key part exists, even if empty (e.g. s3://bucket/)
        raise ValueError(f"Invalid S3 path format: {s3_path}. Must be s3://bucket/key or s3://bucket/.")
    bucket_name = path_parts[0]
    object_key = path_parts[1]
    return bucket_name, object_key

# --- Generic Load/Save with S3 Support ---
def save_to_parquet(df, file_path: str):
    """
    Saves a DataFrame (Polars or Pandas) to a Parquet file, locally or on S3.

    Args:
        df: The DataFrame to save (Polars or Pandas).
        file_path: The full path to save the file to (e.g., /path/to/file.parquet or s3://bucket/key.parquet).
    """
    logger.info(f"Attempting to save DataFrame to: {file_path}")
    if is_s3_path(file_path):
        try:
            bucket_name, object_key = parse_s3_path(file_path)
            if not object_key: # Do not allow saving to just the bucket root without a key name
                 logger.error(f"S3 object key is empty in path: {file_path}. Cannot save without an object name.")
                 raise ValueError(f"S3 object key is empty in path: {file_path}")

            s3_client = boto3.client('s3')
            buffer = io.BytesIO()

            if isinstance(df, pl.DataFrame):
                df.write_parquet(buffer)
            elif isinstance(df, pd.DataFrame):
                df.to_parquet(buffer)
            else:
                logger.error(f"Unsupported DataFrame type: {type(df)}. Must be Polars or Pandas.")
                raise ValueError("Unsupported DataFrame type. Must be Polars or Pandas.")

            buffer.seek(0) # Rewind buffer to the beginning before uploading
            s3_client.put_object(Bucket=bucket_name, Key=object_key, Body=buffer.getvalue())
            logger.info(f"Successfully saved DataFrame to S3: {file_path}")
        except (NoCredentialsError, PartialCredentialsError):
            logger.error(f"AWS credentials not found or incomplete for S3 path: {file_path}", exc_info=True)
            raise
        except ClientError as e:
            logger.error(f"AWS S3 ClientError saving to {file_path}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error saving DataFrame to S3 path {file_path}: {e}", exc_info=True)
            raise
    else: # Local file path
        try:
            # Ensure parent directory exists for local files
            local_path = Path(file_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(df, pl.DataFrame):
                df.write_parquet(local_path)
            elif isinstance(df, pd.DataFrame):
                df.to_parquet(local_path)
            else:
                logger.error(f"Unsupported DataFrame type: {type(df)}. Must be Polars or Pandas.")
                raise ValueError("Unsupported DataFrame type. Must be Polars or Pandas.")
            logger.info(f"Successfully saved DataFrame to local path: {file_path}")
        except Exception as e:
            logger.error(f"Error saving DataFrame to local path {file_path}: {e}", exc_info=True)
            raise

def load_from_parquet(file_path: str, use_polars: bool = True):
    """
    Loads a DataFrame (Polars or Pandas) from a Parquet file, locally or on S3.

    Args:
        file_path: The full path to load the file from (e.g., /path/to/file.parquet or s3://bucket/key.parquet).
        use_polars: If True, loads as Polars DataFrame; otherwise, loads as Pandas DataFrame.

    Returns:
        A Polars or Pandas DataFrame. Raises FileNotFoundError or S3 specific error if loading fails.
    """
    logger.info(f"Attempting to load DataFrame from: {file_path}")
    if is_s3_path(file_path):
        try:
            bucket_name, object_key = parse_s3_path(file_path)
            if not object_key: # Do not allow loading from just the bucket root without a key name
                 logger.error(f"S3 object key is empty in path: {file_path}. Cannot load without an object name.")
                 raise ValueError(f"S3 object key is empty in path: {file_path}")

            s3_client = boto3.client('s3')

            response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
            buffer = io.BytesIO(response['Body'].read())

            if use_polars:
                df_to_load = pl.read_parquet(buffer)
            else:
                df_to_load = pd.read_parquet(buffer)

            logger.info(f"Successfully loaded DataFrame from S3: {file_path}")
            return df_to_load
        except (NoCredentialsError, PartialCredentialsError):
            logger.error(f"AWS credentials not found or incomplete for S3 path: {file_path}", exc_info=True)
            raise
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning(f"S3 object not found at {file_path}: {e}")
                raise FileNotFoundError(f"S3 object not found: {file_path}") from e
            else:
                logger.error(f"AWS S3 ClientError loading from {file_path}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error loading DataFrame from S3 path {file_path}: {e}", exc_info=True)
            raise
    else: # Local file path
        local_path = Path(file_path)
        if not local_path.exists() or not local_path.is_file():
            logger.warning(f"File not found or is not a file at local path {local_path}.")
            raise FileNotFoundError(f"No such file or directory: {local_path}")
        try:
            if use_polars:
                df_to_load = pl.read_parquet(local_path)
            else:
                df_to_load = pd.read_parquet(local_path)
            logger.info(f"Successfully loaded DataFrame from local path: {file_path}")
            return df_to_load
        except Exception as e:
            logger.error(f"Error loading DataFrame from local path {file_path}: {e}", exc_info=True)
            raise


def save_dataframe(df: pl.DataFrame, file_name_with_ext: str, sub_path: str = ""):
    """Saves a Polars DataFrame to a Parquet file relative to BASE_FILE_PATH,
       handling local and S3 base paths."""
    try:
        base_path_str = config.BASE_FILE_PATH
        path_to_use: str

        if is_s3_path(base_path_str):
            # Construct S3 path
            # Normalize base path to ensure it has one trailing slash
            s3_base_normalized = base_path_str.rstrip('/') + '/'

            # Normalize sub_path: remove leading/trailing slashes, ensure it's not empty if it only contained slashes
            clean_sub_path = sub_path.strip('/') if sub_path else ""

            # file_name_with_ext should ideally not have slashes, but strip just in case
            clean_file_name = file_name_with_ext.strip('/')

            # Build the key parts, filtering out empty strings that might result from strip('/')
            key_components = [part for part in [clean_sub_path, clean_file_name] if part]
            full_key = "/".join(key_components)

            # Combine base S3 path with the constructed key
            # Example: s3_base_normalized = "s3://my-bucket/data/"
            # full_key can be "output.parquet" or "results/foo/output.parquet"
            path_to_use = s3_base_normalized + full_key
        else:
            # Construct local path using pathlib
            base_path_obj = Path(base_path_str)
            # sub_path can be multi-level e.g. "results/daily"
            path_to_use = str(base_path_obj / sub_path / file_name_with_ext)

        logger.info(f"Constructed path for save_dataframe: {path_to_use}")
        save_to_parquet(df, path_to_use)

    except AttributeError as e:
        logger.error(f"Configuration error: {e}. Potentially BASE_FILE_PATH not set in config.", exc_info=True)
        raise # Re-raise to indicate failure to the caller
    except Exception as e:
        logger.error(f"Error in save_dataframe for '{file_name_with_ext}' in sub-path '{sub_path}': {e}", exc_info=True)
        raise # Re-raise to indicate failure


def load_dataframe(file_name_with_ext: str, sub_path: str = "") -> pl.DataFrame | None:
    """Loads a Polars DataFrame from a Parquet file using pathlib relative to BASE_FILE_PATH."""
    try:
        base_path_str = config.BASE_FILE_PATH
        path_to_use: str

        if is_s3_path(base_path_str):
            s3_base_normalized = base_path_str.rstrip('/') + '/'
            clean_sub_path = sub_path.strip('/') if sub_path else ""
            clean_file_name = file_name_with_ext.strip('/')
            key_components = [part for part in [clean_sub_path, clean_file_name] if part]
            full_key = "/".join(key_components)
            path_to_use = s3_base_normalized + full_key
        else:
            base_path_obj = Path(base_path_str)
            path_to_use = str(base_path_obj / sub_path / file_name_with_ext)

        logger.info(f"Constructed path for load_dataframe: {path_to_use}")
        # Assuming these older functions are always for Polars.
        return load_from_parquet(path_to_use, use_polars=True)

    except AttributeError as ae:
        logger.error(f"Configuration error: {ae}. Potentially BASE_FILE_PATH not set in config.", exc_info=True)
        return None # Consistent with original behavior on config error
    except FileNotFoundError:
        # Logged by load_from_parquet. This function's original behavior was to return None on file not found.
        return None
    except Exception as e:
        logger.error(f"Error in load_dataframe for '{file_name_with_ext}' in sub-path '{sub_path}': {e}", exc_info=True)
        return None # Consistent with original behavior on other errors


# --- Specific Load/Save Functions (Examples using generic functions) ---

def save_historical_pa_data_with_helpers(df: pl.DataFrame):
    try:
        save_dataframe(df, config.HISTORICAL_PA_HELPERS_FILE)
    except AttributeError: # config.HISTORICAL_PA_HELPERS_FILE might be missing
        logging.exception("HISTORICAL_PA_HELPERS_FILE not found in config.")
    # Other exceptions are caught and logged by save_dataframe


def load_historical_pa_data_with_helpers() -> pl.DataFrame | None:
    try:
        return load_dataframe(config.HISTORICAL_PA_HELPERS_FILE)
    except AttributeError:
        logging.exception("HISTORICAL_PA_HELPERS_FILE not found in config.")
        return None
    # Other exceptions are caught by load_dataframe and result in None


def save_daily_batter_stats(df: pl.DataFrame, date_str: str):
    file_name = f"daily_batter_stats_{date_str}.parquet"
    try:
        save_dataframe(df, file_name, sub_path="daily_features")
    except Exception: # Errors logged by save_dataframe
        pass


def load_daily_batter_stats(date_str: str) -> pl.DataFrame | None:
    file_name = f"daily_batter_stats_{date_str}.parquet"
    try:
        return load_dataframe(file_name, sub_path="daily_features")
    except Exception: # Errors logged by load_dataframe
        return None

def save_simulation_results(df: pl.DataFrame, date_str: str, game_pk: int):
    file_name = f"{game_pk}_sim_probs.parquet"
    # Path object for sub_path construction is fine, it's converted to string before S3 logic
    results_sub_path = Path("results") / date_str
    try:
        save_dataframe(df, file_name, sub_path=str(results_sub_path))
    except Exception: # Errors logged by save_dataframe
        pass


def load_simulation_results_for_date(date_str: str) -> pl.DataFrame | None:
    # This function needs more significant S3 adaptation if results_path_obj is S3
    # as Path().iterdir() won't work.
    # The current load_dataframe can load individual S3 files if given a full S3 path.
    # This function's primary role is to discover files in a "directory".

    try:
        base_path_input = config.BASE_FILE_PATH
        # Construct the "directory" path
        # For local: /base/results/date_str
        # For S3: s3://bucket/prefix/results/date_str/
        dir_path_str: str
        if is_s3_path(base_path_input):
            s3_base_normalized = base_path_input.rstrip('/') + '/'
            results_prefix = "results".strip('/')
            date_prefix = date_str.strip('/')
            # Ensure results_prefix and date_prefix are not empty if they were just "/"
            key_parts = [part for part in [results_prefix, date_prefix] if part]
            dir_path_str = s3_base_normalized + "/".join(key_parts)
            if not dir_path_str.endswith('/'): # Ensure it's a prefix for listing
                dir_path_str += '/'
        else:
            dir_path_str = str(Path(base_path_input) / "results" / date_str)

        logger.info(f"Listing simulation results from directory/prefix: {dir_path_str}")
        all_dfs = []

        if is_s3_path(dir_path_str):
            s3_client = boto3.client('s3')
            bucket_name, prefix = parse_s3_path(dir_path_str) # prefix will include "results/date_str/"

            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

            for page in pages:
                if "Contents" in page:
                    for obj in page['Contents']:
                        object_key = obj['Key']
                        # Ensure it's a parquet file directly under the prefix, not in sub-sub-folders by checking path depth
                        if object_key.endswith(".parquet") and object_key != prefix : # and (object_key.count('/') == prefix.count('/'))
                            s3_file_path = f"s3://{bucket_name}/{object_key}"
                            try:
                                # load_from_parquet used directly as load_dataframe is for predefined sub_path/filename
                                df_temp = load_from_parquet(s3_file_path, use_polars=True)
                                if df_temp is not None: # Should not be None if no error
                                    all_dfs.append(df_temp)
                            except Exception as e: # Catch load_from_parquet errors
                                logger.error(f"Error loading S3 file {s3_file_path}: {e}", exc_info=True)
            if all_dfs:
                logger.info(f"Concatenating {len(all_dfs)} result DataFrames from S3 for date {date_str}.")
                return pl.concat(all_dfs, how="vertical_relaxed")
            logging.warning(f"No simulation result files found or loaded from S3 prefix: {dir_path_str}")
            return None

        else: # Local path
            local_dir_path = Path(dir_path_str)
            if local_dir_path.exists() and local_dir_path.is_dir():
                for file_path_obj in local_dir_path.iterdir():
                    if file_path_obj.is_file() and file_path_obj.suffix == ".parquet":
                        try:
                            # Use load_from_parquet directly for local files too for consistency here
                            df_temp = load_from_parquet(str(file_path_obj), use_polars=True)
                            if df_temp is not None: # Should not be None if no error
                                all_dfs.append(df_temp)
                        except Exception as e: # Catch load_from_parquet errors
                            logger.error(f"Error loading local file {file_path_obj}: {e}", exc_info=True)
                if all_dfs:
                    logger.info(f"Concatenating {len(all_dfs)} local result DataFrames for date {date_str}.")
                    return pl.concat(all_dfs, how="vertical_relaxed")
                logging.warning(f"No simulation result files found or loaded from local directory: {local_dir_path}")
                return None
            else:
                logging.warning(f"Local directory not found or not a directory: {local_dir_path}")
                return None

    except AttributeError as ae: # config.BASE_FILE_PATH missing
        logging.error(f"Configuration error: {ae}. Potentially BASE_FILE_PATH not set in config.", exc_info=True)
        return None
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Error in load_simulation_results_for_date for date {date_str}: {e}", exc_info=True)
        return None

# Removed redundant import comments at the end
