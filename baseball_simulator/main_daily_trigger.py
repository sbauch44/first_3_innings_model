import logging

# import scheduler_service # Hypothetical module to interact with AWS EventBridge etc.
from datetime import date, datetime, timedelta

import config
import data_fetcher
import data_processor
import polars as pl
import pytz  # For timezone handling
import statsapi

# --- Configure Logging ---
# Configure once at the start of your script execution
logging.basicConfig(
    level=logging.INFO,  # Set the minimum level of messages to log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Define log message format
    datefmt="%Y-%m-%d %H:%M:%S",  # Define date/time format
)


def run_incremental_update_and_feature_recalc():
    """
    Fetches yesterday's data, processes it, unions with historical data,
    and recalculates all rolling ballasted stats.
    Saves the updated historical data and the calculated rolling stats.
    """
    logging.info("--- Starting Incremental Update and Feature Recalculation ---")

    # --- 1. Fetch and Process Yesterday's Statcast Data ---
    yesterday = date.today() - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")
    logging.info(f"Fetching Statcast data for: {yesterday_str}")

    try:
        df_new_raw = data_fetcher.fetch_statcast_data()
        if df_new_raw is None or df_new_raw.is_empty():
            logging.info("No new Statcast data found for yesterday.")
            return True  # Indicate success even if no new data
        logging.info(f"Fetched {df_new_raw.shape[0]} new raw rows.")
    except Exception as e:
        logging.error(
            f"Error fetching Statcast data for {yesterday_str}: {e}", exc_info=True
        )  # Log error with traceback
        return False  # Indicate failure

    # Process the new data
    logging.info("Processing new Statcast data...")
    try:
        df_new_pa_outcome = data_processor.process_statcast_data(df_new_raw)
        df_new_pa_helpers = data_processor.create_helper_columns(df_new_pa_outcome)
        logging.info(f"Processed {df_new_pa_helpers.shape[0]} new PA records.")
    except Exception as e:
        logging.error(f"Error processing new Statcast data: {e}", exc_info=True)
        return False

    # --- 2. Load Historical Processed Data ---
    historical_pa_helpers_path = (
        f"{config.BASE_FILE_PATH}historical_pa_data_with_helpers.parquet"
    )
    logging.info(f"Loading historical PA data from: {historical_pa_helpers_path}")
    try:
        df_historical_pa_helpers = pl.read_parquet(historical_pa_helpers_path)
        logging.info(
            f"Loaded {df_historical_pa_helpers.shape[0]} historical PA records."
        )
        # --- 3. Union New Data with Historical ---
        df_pa_full_updated = pl.concat(
            [df_historical_pa_helpers, df_new_pa_helpers],
            how="vertical_relaxed",
        ).unique(subset=["game_pk", "at_bat_number"])
    except Exception as e:
        logging.warning(f"Could not load historical data (may be first run): {e}")
        df_pa_full_updated = (
            df_new_pa_helpers  # Start fresh if historical doesn't exist
        )

    logging.info(f"Total combined PA records: {df_pa_full_updated.shape[0]}")
    if df_pa_full_updated.is_empty():
        logging.warning("Combined PA DataFrame is empty, stopping update.")
        return True  # Return True to allow scheduling to proceed, but log warning

    # --- 4. Save Updated Combined PA Data (with helpers) ---
    try:
        logging.info(
            f"Saving updated historical PA data to: {historical_pa_helpers_path}"
        )
        df_pa_full_updated.write_parquet(historical_pa_helpers_path)
    except Exception as e:
        logging.error(f"Error saving updated historical PA data: {e}", exc_info=True)
        return False  # Indicate failure

    # --- 5. Load Pre-Calculated League Averages (from 2021-2022) ---
    # Assumes they are stored in config.LEAGUE_AVG_RATES dictionary
    league_averages_2122 = config.LEAGUE_AVG_RATES
    logging.info("Using pre-calculated 2021-2022 league averages for ballast.")

    # --- 6. Recalculate Rolling Ballasted Stats on FULL Updated Dataset ---
    # This is the computationally intensive part that runs daily
    try:
        logging.info("Recalculating daily aggregates...")
        df_batter_daily = data_processor.calculate_batter_daily_totals(
            df_pa_full_updated
        )
        df_pitcher_daily = data_processor.calculate_pitcher_daily_totals(
            df_pa_full_updated
        )

        logging.info("Recalculating cumulative stats...")
        df_batter_daily = data_processor.calculate_cumulative_batter_stats(
            df_batter_daily
        )
        df_pitcher_daily = data_processor.calculate_cumulative_pitcher_stats(
            df_pitcher_daily
        )

        logging.info("Applying ballast and calculating final rolling stats...")
        df_batter_daily_final = data_processor.calculate_ballasted_batter_stats(
            df_batter_daily,
            lg_avgs=league_averages_2122,
            ballast_weights=config.BALLAST_WEIGHTS,
        )
        df_pitcher_daily_final = data_processor.calculate_ballasted_pitcher_stats(
            df_pitcher_daily,
            lg_avgs=league_averages_2122,
            ballast_weights=config.BALLAST_WEIGHTS,
        )
    except Exception as e:
        logging.error(f"Error during rolling stat recalculation: {e}", exc_info=True)
        return False

    # --- 7. Recombine Daily Stats to original dataframe---
    try:
        logging.info("selecting relevant batter and pitcher columns...")
        batter_cols = data_processor.get_cols_to_join(df_batter_daily_final, "batter")
        pitcher_cols = data_processor.get_cols_to_join(
            df_pitcher_daily_final, "pitcher"
        )

        logging.info("filtering dataframe to only include relevant columns...")
        batter_stats_to_join = data_processor.select_subset_of_cols(
            df_batter_daily_final,
            "batter",
            batter_cols,
        )
        pitcher_stats_to_join = data_processor.select_subset_of_cols(
            df_pitcher_daily_final,
            "pitcher",
            pitcher_cols,
        )

        logging.info("Joining daily stats back to the original dataframe...")
        main_df = data_processor.join_together_final_df(
            df_pa_full_updated, batter_stats_to_join, pitcher_stats_to_join
        )
    except Exception as e:
        logging.error(f"Error during joining daily stats: {e}", exc_info=True)
        return False

    # --- 8. Save the Final Daily Stats ---
    final_stats_path = f"{config.BASE_FILE_PATH}daily_stats_final.parquet"

    try:
        logging.info(f"Saving calculated daily batter stats to: {final_stats_path}")
        main_df.write_parquet(final_stats_path)
    except Exception as e:
        logging.error(f"Error saving calculated daily stats: {e}", exc_info=True)
        return False  # Indicate failure

    # --- 9. Fetch Park Factors ---
    try:
        logging.info("fetching park factors...")
        park_factors_df = data_fetcher.fetch_park_factors()
        if park_factors_df is None or park_factors_df.is_empty():
            logging.info("No new park factors data found for yesterday.")
            return True  # Indicate success even if no new data
        logging.info(f"Fetched {park_factors_df.shape[0]} new raw rows.")
    except Exception as e:
        logging.error(
            f"Error fetching park factors data for {yesterday_str}: {e}", exc_info=True
        )  # Log error with traceback
        return False  # Indicate failure

    # --- 10. Transform and Save Park Factors ---
    final_park_factors_path = f"{config.BASE_FILE_PATH}park_factors.parquet"
    try:
        logging.info("Transforming park factors data...")
        park_factors_df = data_processor.calculate_park_factors(park_factors_df)
        logging.info(
            f"Saving transformed park factors data to: {final_park_factors_path}"
        )
        park_factors_df.write_parquet(final_park_factors_path)
    except Exception as e:
        logging.error(f"Error transforming park factors data: {e}", exc_info=True)
        return False

    # --- 11. Fetch Defensive Stats ---
    defensive_stats_df = None
    try:
        logging.info("Fetching defensive stats...")
        defensive_stats_df = data_fetcher.fetch_defensive_stats(config.END_YEAR)
        if defensive_stats_df is None or defensive_stats_df.is_empty():
            logging.info("No new defensive stats data found for yesterday.")
            return True  # Indicate success even if no new data
        logging.info(f"Fetched {defensive_stats_df.shape[0]} new raw rows.")
    except Exception as e:
        logging.error(
            f"Error fetching defensive stats data for {yesterday_str}: {e}",
            exc_info=True,
        )

    # --- 12. Transform and Save Defensive Stats ---
    final_defensive_stats_path = f"{config.BASE_FILE_PATH}defensive_stats.parquet"
    if defensive_stats_df is None:
        logging.warning(
            "No defensive stats DataFrame to process. Skipping transformation and save."
        )
        return True
    try:
        logging.info("Transforming defensive stats data...")
        defensive_total_innings = data_processor.calculate_defensive_innings_played(
            df_pa_full_updated
        )
        final_defensive_stats = data_processor.calculate_cumulative_defensive_stats(
            defensive_stats_df, defensive_total_innings
        )

        logging.info(
            f"Saving transformed defensive stats data to: {final_defensive_stats_path}"
        )
        final_defensive_stats.write_parquet(final_defensive_stats_path)
    except Exception as e:
        logging.error(f"Error transforming defensive stats data: {e}", exc_info=True)
        return False

    # --- 13. Fetch Fangraphs Projections ---
    try:
        logging.info("Fetching Fangraphs batter projections...")
        fangraphs_bat_proj_df = data_fetcher.fetch_fangraphs_projections("bat")
        if fangraphs_bat_proj_df is None or fangraphs_bat_proj_df.is_empty():
            logging.info("No new Fangraphs batter projections data found.")
            return True  # Indicate success even if no new data
        logging.info(
            f"Fetched {fangraphs_bat_proj_df.shape[0]} new Fangraphs batter projections."
        )
    except Exception as e:
        logging.error(
            f"Error fetching Fangraphs batter projections: {e}", exc_info=True
        )
        return False  # Indicate failure

    try:
        logging.info("Transforming Fangraphs pitching projections data...")
        fangraphs_pit_proj_df = data_fetcher.fetch_fangraphs_projections("pit")
        if fangraphs_pit_proj_df is None or fangraphs_pit_proj_df.is_empty():
            logging.info("No new Fangraphs pitching projections data found.")
            return True
        logging.info(
            f"Fetched {fangraphs_pit_proj_df.shape[0]} new Fangraphs pitching projections."
        )
    except Exception as e:
        logging.error(
            f"Error fetching Fangraphs pitching projections: {e}", exc_info=True
        )
        return False

    # --- 14. Store Fangraphs Projections ---
    final_bat_projections_path = (
        f"{config.BASE_FILE_PATH}fangraphs_bat_projections.parquet"
    )
    try:
        logging.info(
            f"Saving fangraphs batter projections to: {final_bat_projections_path}"
        )
        formatted_bat_projections = data_processor.process_fangraphs_batter_projections(
            fangraphs_bat_proj_df
        )
        formatted_bat_projections.write_parquet(final_bat_projections_path)
    except Exception as e:
        logging.error(f"Error saving fangraphs batter projections: {e}", exc_info=True)
        return False

    final_pit_projections_path = (
        f"{config.BASE_FILE_PATH}fangraphs_pit_projections.parquet"
    )
    try:
        logging.info(
            f"Saving fangraphs pitcher projections to: {final_pit_projections_path}"
        )
        formatted_pit_projections = (
            data_processor.process_fangraphs_pitcher_projections(
                fangraphs_pit_proj_df,
                use_advanced_imputation=True,  # Start with simple method
            )
        )
        formatted_pit_projections.write_parquet(final_pit_projections_path)
    except Exception as e:
        logging.error(f"Error saving fangraphs pitcher projections: {e}", exc_info=True)
        return False

    logging.info("--- Incremental Update and Feature Recalculation Complete ---")
    return True  # Indicate success


def run_daily_scheduling():
    """Fetches today's schedule and schedules the pre-game simulation triggers."""
    logging.info("\n--- Starting Daily Scheduling ---")
    today = date.today()
    today_str = today.strftime("%Y-%m-%d")

    # 1. Get Daily Schedule
    try:
        # Fetch schedule using statsapi or your chosen method
        schedule = statsapi.schedule(date=today_str, sportId=1)
        if not schedule:
            logging.info("No games scheduled for today.")
            return
        logging.info(f"Found {len(schedule)} games.")
    except Exception as e:
        logging.error(f"Error fetching schedule from statsapi: {e}", exc_info=True)
        return

    # 2. Schedule Per-Game Triggers
    logging.info("Scheduling pre-game triggers...")
    scheduled_count = 0
    skipped_count = 0
    for game in schedule:
        try:
            game_pk = game.get("game_id")
            game_start_str = game.get("game_datetime")
            status = game.get("status", "Unknown").lower()

            # Basic validation
            if not game_pk or not game_start_str or not isinstance(game_pk, int):
                logging.warning(
                    f"  Skipping game due to missing pk or start time: {game}"
                )
                skipped_count += 1
                continue
            if (
                "tbd" in status
                or "postponed" in status
                or "cancelled" in status
                or "suspended" in status
            ):
                logging.info(f"  Skipping game {game_pk} due to status: {status}")
                skipped_count += 1
                continue

            # Convert to datetime and calculate trigger time
            # Ensure timezone awareness - assumes fetched time is UTC ('Z')
            game_start_utc = datetime.fromisoformat(
                game_start_str.replace("Z", "+00:00")
            ).replace(tzinfo=pytz.utc)
            trigger_time_utc = game_start_utc - timedelta(minutes=55)

            # Don't schedule tasks in the past
            if trigger_time_utc < datetime.now(pytz.utc):
                logging.info(
                    f"  Skipping game {game_pk} - Trigger time {trigger_time_utc} is in the past."
                )
                skipped_count += 1
                continue

            logging.info(
                f"  Scheduling trigger for game {game_pk} at {trigger_time_utc} UTC"
            )
            scheduled_count += 1

            # --- Replace with actual scheduling service call (e.g., AWS EventBridge put_rule/schedule) ---
            # Example hypothetical call:
            # success = scheduler_service.schedule_pre_game_task(
            #     schedule_time=trigger_time_utc,
            #     game_pk=game_pk,
            #     target_arn=config.PRE_GAME_TARGET_ARN
            # )
            # if not success:
            #    logging.error(f"   Failed to schedule trigger for game {game_pk}")
            #    scheduled_count -= 1 # Decrement if failed
            # --------------------------------------------------------------------------------------

        except Exception as e:
            logging.error(
                f"Error processing/scheduling game {game.get('game_id', 'N/A')}: {e}",
                exc_info=True,
            )
            skipped_count += 1

    logging.info(
        f"Attempted to schedule triggers for {scheduled_count} games. Skipped {skipped_count} games."
    )
    logging.info("\n--- Daily Scheduling Complete ---")


# --- Main Execution Block ---
if __name__ == "__main__":
    # Configure logging when script starts
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info(f"Starting daily process at {datetime.now()}...")

    # Step 1: Update historical data and recalculate features
    update_success = run_incremental_update_and_feature_recalc()

    # Step 2: If update was successful, schedule today's tasks
    if update_success:
        run_daily_scheduling()
    else:
        logging.error(
            "Halting daily process due to error during data update/recalculation."
        )

    logging.info(f"Daily process finished at {datetime.now()}.")
