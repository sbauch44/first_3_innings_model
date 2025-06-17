import logging
# Removed subprocess, sys
from datetime import date, datetime, timedelta # Ensure all are imported
from pathlib import Path # Keep Path for now, might be used by config or new logic

import config
import data_fetcher
import data_processor
import polars as pl # Keep polars for data_processor return types etc.
import pytz # Keep pytz for timezone handling
import statsapi
# Removed APScheduler imports:
# from apscheduler.executors.pool import ThreadPoolExecutor
# from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
# from apscheduler.schedulers.blocking import BlockingScheduler
from baseball_simulator import storage

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_incremental_update_and_feature_recalc():
    """
    Performs daily incremental update of Statcast data, feature recalculation,
    and updates other relevant data like park factors and projections.
    Uses S3-aware storage functions for Parquet I/O.
    """
    logger.info("--- Starting Incremental Update and Feature Recalculation ---")

    # --- 1. Fetch and Process Yesterday's Statcast Data ---
    yesterday = date.today() - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")
    logger.info(f"Fetching Statcast data for: {yesterday_str}")

    try:
        df_new_raw = data_fetcher.fetch_statcast_data()
        if df_new_raw is None or df_new_raw.is_empty():
            logger.info("No new Statcast data found for yesterday.")
            return True
        logger.info(f"Fetched {df_new_raw.shape[0]} new raw rows.")
    except Exception as e:
        logger.error(
            f"Error fetching Statcast data for {yesterday_str}: {e}", exc_info=True
        )
        return False

    logger.info("Processing new Statcast data...")
    try:
        df_new_pa_outcome = data_processor.process_statcast_data(df_new_raw)
        df_new_pa_helpers = data_processor.create_helper_columns(df_new_pa_outcome)
        logger.info(f"Processed {df_new_pa_helpers.shape[0]} new PA records.")
    except Exception as e:
        logger.error(f"Error processing new Statcast data: {e}", exc_info=True)
        return False

    # --- 2. Load Historical Processed Data ---
    historical_pa_helpers_path = (
        f"{config.BASE_FILE_PATH}historical_pa_data_with_helpers.parquet"
    )
    logger.info(f"Loading historical PA data from: {historical_pa_helpers_path}")
    try:
        df_historical_pa_helpers = storage.load_from_parquet(historical_pa_helpers_path, use_polars=True)
        if df_historical_pa_helpers is None:
             raise FileNotFoundError("load_from_parquet returned None for historical_pa_helpers_path")
        logger.info(
            f"Loaded {df_historical_pa_helpers.shape[0]} historical PA records."
        )
        df_pa_full_updated = pl.concat(
            [df_historical_pa_helpers, df_new_pa_helpers],
            how="vertical_relaxed",
        ).unique(subset=["game_pk", "at_bat_number"])
    except FileNotFoundError as e:
        logger.warning(f"Historical data file not found (may be first run): {e}")
        df_pa_full_updated = df_new_pa_helpers
    except Exception as e:
        logger.warning(f"Could not load or process historical data (may be first run): {e}", exc_info=True)
        df_pa_full_updated = df_new_pa_helpers

    logger.info(f"Total combined PA records: {df_pa_full_updated.shape[0]}")
    if df_pa_full_updated.is_empty():
        logger.warning("Combined PA DataFrame is empty, stopping update.")
        return True

    # --- 4. Save Updated Combined PA Data (with helpers) ---
    try:
        logger.info(
            f"Saving updated historical PA data to: {historical_pa_helpers_path}"
        )
        storage.save_to_parquet(df_pa_full_updated, historical_pa_helpers_path)
    except Exception as e:
        logger.error(f"Error saving updated historical PA data: {e}", exc_info=True)
        return False

    league_averages_2122 = config.LEAGUE_AVG_RATES
    logger.info("Using pre-calculated 2021-2022 league averages for ballast.")

    try:
        logger.info("Recalculating daily aggregates...")
        df_batter_daily = data_processor.calculate_batter_daily_totals(df_pa_full_updated)
        df_pitcher_daily = data_processor.calculate_pitcher_daily_totals(df_pa_full_updated)
        logger.info("Recalculating cumulative stats...")
        df_batter_daily = data_processor.calculate_cumulative_batter_stats(df_batter_daily)
        df_pitcher_daily = data_processor.calculate_cumulative_pitcher_stats(df_pitcher_daily)
        logger.info("Applying ballast and calculating final rolling stats...")
        df_batter_daily_final = data_processor.calculate_ballasted_batter_stats(
            df_batter_daily, lg_avgs=league_averages_2122, ballast_weights=config.BALLAST_WEIGHTS,
        )
        df_pitcher_daily_final = data_processor.calculate_ballasted_pitcher_stats(
            df_pitcher_daily, lg_avgs=league_averages_2122, ballast_weights=config.BALLAST_WEIGHTS,
        )
    except Exception as e:
        logger.error(f"Error during rolling stat recalculation: {e}", exc_info=True)
        return False

    try:
        logger.info("selecting relevant batter and pitcher columns...")
        batter_cols = data_processor.get_cols_to_join(df_batter_daily_final, "batter")
        pitcher_cols = data_processor.get_cols_to_join(df_pitcher_daily_final, "pitcher")
        logger.info("filtering dataframe to only include relevant columns...")
        batter_stats_to_join = data_processor.select_subset_of_cols(df_batter_daily_final, "batter", batter_cols)
        pitcher_stats_to_join = data_processor.select_subset_of_cols(df_pitcher_daily_final, "pitcher", pitcher_cols)
        logger.info("Joining daily stats back to the original dataframe...")
        main_df = data_processor.join_together_final_df(
            df_pa_full_updated, batter_stats_to_join, pitcher_stats_to_join
        )
    except Exception as e:
        logger.error(f"Error during joining daily stats: {e}", exc_info=True)
        return False

    final_stats_path = f"{config.BASE_FILE_PATH}daily_stats_final.parquet"
    try:
        logger.info(f"Saving calculated daily stats to: {final_stats_path}")
        storage.save_to_parquet(main_df, final_stats_path)
    except Exception as e:
        logger.error(f"Error saving calculated daily stats: {e}", exc_info=True)
        return False

    try:
        logger.info("fetching park factors...")
        park_factors_df = data_fetcher.fetch_park_factors()
        if park_factors_df is None or park_factors_df.is_empty():
            logger.info("No new park factors data returned from fetcher.")
    except Exception as e:
        logger.error(f"Error fetching park factors data: {e}", exc_info=True)
        park_factors_df = None # Ensure it's None if fetch fails

    final_park_factors_path = f"{config.BASE_FILE_PATH}park_factors.parquet"
    if park_factors_df is not None and not park_factors_df.is_empty():
        try:
            logger.info("Transforming park factors data...")
            park_factors_df = data_processor.calculate_park_factors(park_factors_df)
            logger.info(f"Saving transformed park factors data to: {final_park_factors_path}")
            storage.save_to_parquet(park_factors_df, final_park_factors_path)
        except Exception as e:
            logger.error(f"Error transforming or saving park factors data: {e}", exc_info=True)
            # Not returning False, as park factors might be considered non-critical for the main flow
    else:
        logger.info("Skipping park factors transformation and save due to no or error in data.")

    defensive_stats_df = None
    try:
        logger.info("Fetching defensive stats...")
        defensive_stats_df = data_fetcher.fetch_defensive_stats(config.END_YEAR)
        if defensive_stats_df is None or defensive_stats_df.is_empty():
            logger.info("No new defensive stats data found.")
    except Exception as e:
        logger.error(f"Error fetching defensive stats data: {e}", exc_info=True)
        defensive_stats_df = None # Ensure it's None

    final_defensive_stats_path = f"{config.BASE_FILE_PATH}defensive_stats.parquet"
    if defensive_stats_df is not None and not defensive_stats_df.is_empty():
        try:
            logger.info("Transforming defensive stats data...")
            defensive_total_innings = data_processor.calculate_defensive_innings_played(df_pa_full_updated)
            final_defensive_stats = data_processor.calculate_cumulative_defensive_stats(
                defensive_stats_df, defensive_total_innings
            )
            logger.info(f"Saving transformed defensive stats data to: {final_defensive_stats_path}")
            storage.save_to_parquet(final_defensive_stats, final_defensive_stats_path)
        except Exception as e:
            logger.error(f"Error transforming or saving defensive stats data: {e}", exc_info=True)
            # Not returning False, as defensive stats might be non-critical
    else:
        logger.info("Skipping defensive stats transformation and save due to no or error in data.")

    fangraphs_bat_proj_df = None
    try:
        logger.info("Fetching Fangraphs batter projections...")
        fangraphs_bat_proj_df = data_fetcher.fetch_fangraphs_projections("bat")
        if fangraphs_bat_proj_df is None or fangraphs_bat_proj_df.is_empty():
            logger.info("No new Fangraphs batter projections data found.")
    except Exception as e:
        logger.error(f"Error fetching Fangraphs batter projections: {e}", exc_info=True)
        fangraphs_bat_proj_df = None

    final_bat_projections_path = f"{config.BASE_FILE_PATH}fangraphs_bat_projections.parquet"
    if fangraphs_bat_proj_df is not None and not fangraphs_bat_proj_df.is_empty():
        try:
            logger.info(f"Saving fangraphs batter projections to: {final_bat_projections_path}")
            formatted_bat_projections = data_processor.process_fangraphs_batter_projections(fangraphs_bat_proj_df)
            storage.save_to_parquet(formatted_bat_projections, final_bat_projections_path)
        except Exception as e:
            logger.error(f"Error saving fangraphs batter projections: {e}", exc_info=True)
    else:
        logger.info("Skipping Fangraphs batter projections save due to no or error in data.")

    fangraphs_pit_proj_df = None
    try:
        logger.info("Fetching Fangraphs pitching projections...")
        fangraphs_pit_proj_df = data_fetcher.fetch_fangraphs_projections("pit")
        if fangraphs_pit_proj_df is None or fangraphs_pit_proj_df.is_empty():
            logger.info("No new Fangraphs pitching projections data found.")
    except Exception as e:
        logger.error(f"Error fetching Fangraphs pitching projections: {e}", exc_info=True)
        fangraphs_pit_proj_df = None

    final_pit_projections_path = f"{config.BASE_FILE_PATH}fangraphs_pit_projections.parquet"
    if fangraphs_pit_proj_df is not None and not fangraphs_pit_proj_df.is_empty():
        try:
            logger.info("Saving fangraphs pitcher projections to: %s", final_pit_projections_path)
            formatted_pit_projections = data_processor.process_fangraphs_pitcher_projections(
                fangraphs_pit_proj_df, use_advanced_imputation=True
            )
            storage.save_to_parquet(formatted_pit_projections, final_pit_projections_path)
        except Exception as e:
            logger.error(f"Error saving fangraphs pitcher projections: {e}", exc_info=True)
    else:
        logger.info("Skipping Fangraphs pitcher projections save due to no or error in data.")

    logger.info("--- Incremental Update and Feature Recalculation Complete ---")
    return True


# Main execution
if __name__ == "__main__":
    logger.info(f"Starting daily process at {datetime.now(tz=pytz.utc)}...")

    update_success = run_incremental_update_and_feature_recalc()

    if update_success:
        logger.info("--- Identifying Games for Pre-Game Simulation ---")
        today = date.today()
        today_str = today.strftime("%Y-%m-%d")
        try:
            games_schedule = statsapi.schedule(date=today_str, sportId=1)
            if not games_schedule:
                logger.info("No games scheduled for today.")
            else:
                logger.info(f"Found {len(games_schedule)} games in schedule for today.")
                game_pks_to_simulate = []
                for game in games_schedule:
                    game_pk = game.get("game_id")
                    status = game.get("status", "Unknown").lower()
                    game_start_str = game.get("game_datetime") # For potential future filtering

                    if not game_pk or not game_start_str:
                        logger.warning(f"Skipping game due to missing game_id or game_datetime: {game}")
                        continue

                    # Filter out games that are not suitable for simulation
                    # Added "completed", "final", "game over" to status checks
                    if any(word in status for word in ["tbd", "postponed", "cancelled", "suspended", "completed", "final", "game over"]):
                        logger.info(f"Skipping game {game_pk} due to status: {status}")
                        continue

                    # Optional: Add time-based filtering if needed
                    # game_start_utc = datetime.fromisoformat(game_start_str)
                    # if game_start_utc.tzinfo is None: # Ensure timezone aware
                    #    game_start_utc = pytz.utc.localize(game_start_utc)
                    # # Example: Skip games that started more than 3 hours ago or are too far in the future
                    # if game_start_utc < datetime.now(pytz.utc) - timedelta(hours=3):
                    #    logger.info(f"Skipping game {game_pk} as it likely already started or is over ({game_start_str}).")
                    #    continue
                    # if game_start_utc > datetime.now(pytz.utc) + timedelta(days=1): # Example: more than 1 day out
                    #    logger.info(f"Skipping game {game_pk} as it is too far in the future ({game_start_str}).")
                    #    continue

                    game_pks_to_simulate.append(game_pk)

                if game_pks_to_simulate:
                    logger.info(f"Identified {len(game_pks_to_simulate)} games to be scheduled for simulation:")
                    for pk in game_pks_to_simulate:
                        # This specific log format can be used by downstream processes
                        logger.info(f"SCHEDULING_PRE_GAME_SIMULATION_FOR_GAME_PK: {pk}")
                else:
                    logger.info("No suitable games found to schedule for simulation today.")

        except Exception as e:
            logger.error(f"Error fetching or processing game schedule for simulation triggers: {e}", exc_info=True)
    else:
        logger.warning("Data update and feature recalculation failed. Skipping game scheduling identification.")

    logger.info(f"Daily process finished at {datetime.now(tz=pytz.utc)}.")
