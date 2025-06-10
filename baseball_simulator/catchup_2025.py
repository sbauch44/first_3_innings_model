import logging
from datetime import date, datetime

import config
import data_fetcher
import data_processor
import polars as pl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def catch_up_2025_data():
    """Fetch all 2025 Statcast data to date and process it."""

    start_date = "2025-01-01"
    end_date = date.today().strftime("%Y-%m-%d")

    logger.info(f"Fetching ALL Statcast data from {start_date} to {end_date}...")

    # 1. Fetch all 2025 data
    try:
        df_all_2025_raw = data_fetcher.fetch_statcast_data(start_date, end_date)
        logger.info(f"Fetched {df_all_2025_raw.shape[0]} raw rows for 2025")
    except Exception as e:
        logger.error(f"Error fetching 2025 data: {e}")
        return False

    # 2. Process the data
    logger.info("Processing 2025 Statcast data...")
    df_2025_pa_outcome = data_processor.process_statcast_data(df_all_2025_raw)
    df_2025_pa_helpers = data_processor.create_helper_columns(df_2025_pa_outcome)

    # 3. Load any existing historical data (2021-2024)
    historical_pa_helpers_path = (
        f"{config.BASE_FILE_PATH}historical_pa_data_with_helpers.parquet"
    )
    try:
        df_historical = pl.read_parquet(historical_pa_helpers_path)
        logger.info(f"Loaded {df_historical.shape[0]} historical records")

        # Combine historical + 2025 data
        df_full_updated = pl.concat(
            [df_historical, df_2025_pa_helpers], how="vertical_relaxed"
        )
        df_full_updated = df_full_updated.unique(subset=["game_pk", "at_bat_number"])

    except Exception as e:
        logger.warning(f"No historical data found: {e}. Using only 2025 data.")
        df_full_updated = df_2025_pa_helpers

    logger.info(f"Total combined records: {df_full_updated.shape[0]}")

    # 4. Save updated historical data
    df_full_updated.write_parquet(historical_pa_helpers_path)
    logger.info("Saved updated historical data")

    # 5. Continue with normal daily processing
    logger.info("Running full recalculation...")
    # ... rest of your normal daily processing pipeline

    return True


if __name__ == "__main__":
    catch_up_2025_data()
