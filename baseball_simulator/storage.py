# storage.py
import logging
from pathlib import Path  # Import Path from pathlib

import polars as pl

import config  # Assuming config.py is in the same directory

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")

# --- Generic Load/Save ---
def save_dataframe(df: pl.DataFrame, file_name_with_ext: str, sub_path: str = ""):
    """Saves a Polars DataFrame to a Parquet file using pathlib."""
    full_path_obj = None
    try:
        base_path_obj = Path(config.BASE_FILE_PATH)
        # Construct full path using the / operator for Path objects
        full_path_obj = base_path_obj / sub_path / file_name_with_ext

        # Ensure parent directory exists
        full_path_obj.parent.mkdir(parents=True, exist_ok=True)

        df.write_parquet(full_path_obj) # Polars write_parquet accepts Path objects
        logging.info(f"Successfully saved DataFrame to {full_path_obj}")
    except AttributeError as ae:
        logging.error(f"Configuration error (e.g., BASE_FILE_PATH not set in config): {ae}", exc_info=True)
    except Exception as e:
        logging.error(f"Error saving DataFrame to {full_path_obj if full_path_obj else file_name_with_ext}: {e}", exc_info=True)


def load_dataframe(file_name_with_ext: str, sub_path: str = "") -> pl.DataFrame | None:
    """Loads a Polars DataFrame from a Parquet file using pathlib."""
    try:
        base_path_obj = Path(config.BASE_FILE_PATH)
        full_path_obj = base_path_obj / sub_path / file_name_with_ext
    except AttributeError as ae:
        logging.error(f"Configuration error (e.g., BASE_FILE_PATH not set in config): {ae}", exc_info=True)
        return None

    logging.info(f"Attempting to load DataFrame from: {full_path_obj}")
    if not full_path_obj.exists() or not full_path_obj.is_file():
        logging.warning(f"File not found or is not a file at {full_path_obj}. Returning None.")
        return None
    try:
        df = pl.read_parquet(full_path_obj) # Polars read_parquet accepts Path objects
        logging.info(f"Successfully loaded DataFrame from {full_path_obj}")
        return df
    except Exception as e:
        logging.error(f"Error loading DataFrame from {full_path_obj}: {e}", exc_info=True)
        return None

# --- Specific Load/Save Functions (Examples using generic functions) ---

def save_historical_pa_data_with_helpers(df: pl.DataFrame):
    try:
        save_dataframe(df, config.HISTORICAL_PA_HELPERS_FILE)
    except AttributeError:
        logging.exception("HISTORICAL_PA_HELPERS_FILE not found in config.")


def load_historical_pa_data_with_helpers() -> pl.DataFrame | None:
    try:
        return load_dataframe(config.HISTORICAL_PA_HELPERS_FILE)
    except AttributeError:
        logging.exception("HISTORICAL_PA_HELPERS_FILE not found in config.")
        return None


def save_daily_batter_stats(df: pl.DataFrame, date_str: str):
    file_name = f"daily_batter_stats_{date_str}.parquet" # Example naming
    save_dataframe(df, file_name, sub_path="daily_features")


def load_daily_batter_stats(date_str: str) -> pl.DataFrame | None:
    file_name = f"daily_batter_stats_{date_str}.parquet"
    return load_dataframe(file_name, sub_path="daily_features")

# Add similar specific functions for pitcher stats, final simulation results, etc.

def save_simulation_results(df: pl.DataFrame, date_str: str, game_pk: int):
    file_name = f"{game_pk}_sim_probs.parquet"
    # Construct sub_path using Path's / operator for platform independence
    results_sub_path = Path("results") / date_str
    save_dataframe(df, file_name, sub_path=str(results_sub_path)) # Convert Path to str for sub_path arg

def load_simulation_results_for_date(date_str: str) -> pl.DataFrame | None:
    try:
        base_path_obj = Path(config.BASE_FILE_PATH)
    except AttributeError:
        logging.exception("BASE_FILE_PATH not found in config.")
        return None

    results_path_obj = base_path_obj / "results" / date_str
    all_dfs = []

    if results_path_obj.exists() and results_path_obj.is_dir():
        logging.info(f"Loading simulation results from directory: {results_path_obj}")
        for file_path_obj in results_path_obj.iterdir(): # Iterate over Path objects
            if file_path_obj.is_file() and file_path_obj.suffix == ".parquet":
                # Pass relative sub_path and file name to load_dataframe
                relative_sub_path = results_path_obj.relative_to(base_path_obj)
                df_temp = load_dataframe(file_path_obj.name, sub_path=str(relative_sub_path))
                if df_temp is not None:
                    all_dfs.append(df_temp)
        if all_dfs:
            logging.info(f"Concatenating {len(all_dfs)} result DataFrames for date {date_str}.")
            return pl.concat(all_dfs, how="vertical_relaxed")
    logging.warning(f"No simulation result files found in directory: {results_path_obj}")
    return None
