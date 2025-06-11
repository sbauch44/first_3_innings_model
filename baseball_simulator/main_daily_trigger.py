import logging
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import config
import data_fetcher
import data_processor
import polars as pl
import pytz
import statsapi
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.blocking import BlockingScheduler

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_incremental_update_and_feature_recalc():
    """
    Your existing function - keeping it unchanged
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

    # Process the new data
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
        df_historical_pa_helpers = pl.read_parquet(historical_pa_helpers_path)
        logger.info(
            f"Loaded {df_historical_pa_helpers.shape[0]} historical PA records."
        )
        # --- 3. Union New Data with Historical ---
        df_pa_full_updated = pl.concat(
            [df_historical_pa_helpers, df_new_pa_helpers],
            how="vertical_relaxed",
        ).unique(subset=["game_pk", "at_bat_number"])
    except Exception as e:
        logger.warning(f"Could not load historical data (may be first run): {e}")
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
        df_pa_full_updated.write_parquet(historical_pa_helpers_path)
    except Exception as e:
        logger.error(f"Error saving updated historical PA data: {e}", exc_info=True)
        return False

    # --- 5. Load Pre-Calculated League Averages (from 2021-2022) ---
    league_averages_2122 = config.LEAGUE_AVG_RATES
    logger.info("Using pre-calculated 2021-2022 league averages for ballast.")

    # --- 6. Recalculate Rolling Ballasted Stats on FULL Updated Dataset ---
    try:
        logger.info("Recalculating daily aggregates...")
        df_batter_daily = data_processor.calculate_batter_daily_totals(
            df_pa_full_updated
        )
        df_pitcher_daily = data_processor.calculate_pitcher_daily_totals(
            df_pa_full_updated
        )

        logger.info("Recalculating cumulative stats...")
        df_batter_daily = data_processor.calculate_cumulative_batter_stats(
            df_batter_daily
        )
        df_pitcher_daily = data_processor.calculate_cumulative_pitcher_stats(
            df_pitcher_daily
        )

        logger.info("Applying ballast and calculating final rolling stats...")
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
        logger.error(f"Error during rolling stat recalculation: {e}", exc_info=True)
        return False

    # --- 7. Recombine Daily Stats to original dataframe---
    try:
        logger.info("selecting relevant batter and pitcher columns...")
        batter_cols = data_processor.get_cols_to_join(df_batter_daily_final, "batter")
        pitcher_cols = data_processor.get_cols_to_join(
            df_pitcher_daily_final, "pitcher"
        )

        logger.info("filtering dataframe to only include relevant columns...")
        batter_stats_to_join = data_processor.select_subset_of_cols(
            df_batter_daily_final, "batter", batter_cols
        )
        pitcher_stats_to_join = data_processor.select_subset_of_cols(
            df_pitcher_daily_final, "pitcher", pitcher_cols
        )

        logger.info("Joining daily stats back to the original dataframe...")
        main_df = data_processor.join_together_final_df(
            df_pa_full_updated, batter_stats_to_join, pitcher_stats_to_join
        )
    except Exception as e:
        logger.error(f"Error during joining daily stats: {e}", exc_info=True)
        return False

    # --- 8. Save the Final Daily Stats ---
    final_stats_path = f"{config.BASE_FILE_PATH}daily_stats_final.parquet"
    try:
        logger.info(f"Saving calculated daily batter stats to: {final_stats_path}")
        main_df.write_parquet(final_stats_path)
    except Exception as e:
        logger.error(f"Error saving calculated daily stats: {e}", exc_info=True)
        return False

    # --- 9. Fetch Park Factors ---
    try:
        logger.info("fetching park factors...")
        park_factors_df = data_fetcher.fetch_park_factors()
        if park_factors_df is None or park_factors_df.is_empty():
            logger.info("No new park factors data found for yesterday.")
            return True
        logger.info(f"Fetched {park_factors_df.shape[0]} new raw rows.")
    except Exception as e:
        logger.error(
            f"Error fetching park factors data for {yesterday_str}: {e}", exc_info=True
        )
        return False

    # --- 10. Transform and Save Park Factors ---
    final_park_factors_path = f"{config.BASE_FILE_PATH}park_factors.parquet"
    try:
        logger.info("Transforming park factors data...")
        park_factors_df = data_processor.calculate_park_factors(park_factors_df)
        logger.info(
            f"Saving transformed park factors data to: {final_park_factors_path}"
        )
        park_factors_df.write_parquet(final_park_factors_path)
    except Exception as e:
        logger.error(f"Error transforming park factors data: {e}", exc_info=True)
        return False

    # --- 11. Fetch Defensive Stats ---
    defensive_stats_df = None
    try:
        logger.info("Fetching defensive stats...")
        defensive_stats_df = data_fetcher.fetch_defensive_stats(config.END_YEAR)
        if defensive_stats_df is None or defensive_stats_df.is_empty():
            logger.info("No new defensive stats data found for yesterday.")
            return True
        logger.info(f"Fetched {defensive_stats_df.shape[0]} new raw rows.")
    except Exception as e:
        logger.error(
            f"Error fetching defensive stats data for {yesterday_str}: {e}",
            exc_info=True,
        )

    # --- 12. Transform and Save Defensive Stats ---
    final_defensive_stats_path = f"{config.BASE_FILE_PATH}defensive_stats.parquet"
    if defensive_stats_df is None:
        logger.warning(
            "No defensive stats DataFrame to process. Skipping transformation and save."
        )
        return True
    try:
        logger.info("Transforming defensive stats data...")
        defensive_total_innings = data_processor.calculate_defensive_innings_played(
            df_pa_full_updated
        )
        final_defensive_stats = data_processor.calculate_cumulative_defensive_stats(
            defensive_stats_df, defensive_total_innings
        )
        logger.info(
            f"Saving transformed defensive stats data to: {final_defensive_stats_path}"
        )
        final_defensive_stats.write_parquet(final_defensive_stats_path)
    except Exception as e:
        logger.error(f"Error transforming defensive stats data: {e}", exc_info=True)
        return False

    # --- 13. Fetch Fangraphs Projections ---
    try:
        logger.info("Fetching Fangraphs batter projections...")
        fangraphs_bat_proj_df = data_fetcher.fetch_fangraphs_projections("bat")
        if fangraphs_bat_proj_df is None or fangraphs_bat_proj_df.is_empty():
            logger.info("No new Fangraphs batter projections data found.")
            return True
        logger.info(
            f"Fetched {fangraphs_bat_proj_df.shape[0]} new Fangraphs batter projections."
        )
    except Exception as e:
        logger.error(f"Error fetching Fangraphs batter projections: {e}", exc_info=True)
        return False

    try:
        logger.info("Transforming Fangraphs pitching projections data...")
        fangraphs_pit_proj_df = data_fetcher.fetch_fangraphs_projections("pit")
        if fangraphs_pit_proj_df is None or fangraphs_pit_proj_df.is_empty():
            logger.info("No new Fangraphs pitching projections data found.")
            return True
        logger.info(
            f"Fetched {fangraphs_pit_proj_df.shape[0]} new Fangraphs pitching projections."
        )
    except Exception as e:
        logger.error(
            f"Error fetching Fangraphs pitching projections: {e}", exc_info=True
        )
        return False

    # --- 14. Store Fangraphs Projections ---
    final_bat_projections_path = (
        f"{config.BASE_FILE_PATH}fangraphs_bat_projections.parquet"
    )
    try:
        logger.info(
            f"Saving fangraphs batter projections to: {final_bat_projections_path}"
        )
        formatted_bat_projections = data_processor.process_fangraphs_batter_projections(
            fangraphs_bat_proj_df
        )
        formatted_bat_projections.write_parquet(final_bat_projections_path)
    except Exception as e:
        logger.error(f"Error saving fangraphs batter projections: {e}", exc_info=True)
        return False

    final_pit_projections_path = (
        f"{config.BASE_FILE_PATH}fangraphs_pit_projections.parquet"
    )
    try:
        logger.info(
            "Saving fangraphs pitcher projections to: %s", final_pit_projections_path
        )
        formatted_pit_projections = (
            data_processor.process_fangraphs_pitcher_projections(
                fangraphs_pit_proj_df, use_advanced_imputation=True
            )
        )
        formatted_pit_projections.write_parquet(final_pit_projections_path)
    except Exception as e:
        logger.error(f"Error saving fangraphs pitcher projections: {e}", exc_info=True)
        return False

    logger.info("--- Incremental Update and Feature Recalculation Complete ---")
    return True


def create_windows_scheduled_tasks_for_games():
    """
    Create individual Windows scheduled tasks for each game instead of using APScheduler
    This approach is more reliable on Windows
    """
    logger.info("--- Creating Windows Scheduled Tasks for Today's Games ---")

    today = date.today()
    today_str = today.strftime("%Y-%m-%d")

    try:
        # Get today's schedule
        schedule = statsapi.schedule(date=today_str, sportId=1)
        if not schedule:
            logger.info("No games scheduled for today.")
            return True

        logger.info(f"Found {len(schedule)} games.")

        # Get paths
        script_dir = Path(__file__).parent.absolute()
        python_exe = sys.executable
        pre_game_script = script_dir / "main_pre_game_trigger.py"

        scheduled_count = 0
        skipped_count = 0

        for game in schedule:
            try:
                game_pk = game.get("game_id")
                game_start_str = game.get("game_datetime")
                status = game.get("status", "Unknown").lower()

                # Validation
                if not game_pk or not game_start_str:
                    logger.warning(f"Skipping game due to missing data: {game}")
                    skipped_count += 1
                    continue

                if any(
                    word in status
                    for word in ["tbd", "postponed", "cancelled", "suspended"]
                ):
                    logger.info(f"Skipping game {game_pk} due to status: {status}")
                    skipped_count += 1
                    continue

                # Calculate trigger time
                game_start_utc = datetime.fromisoformat(game_start_str)
                if game_start_utc.tzinfo is None:
                    game_start_utc = pytz.utc.localize(game_start_utc)

                # Convert to local time for Windows Task Scheduler
                local_tz = pytz.timezone("America/New_York")  # Adjust to your timezone
                game_start_local = game_start_utc.astimezone(local_tz)
                trigger_time_local = game_start_local - timedelta(minutes=55)

                # Don't schedule in the past
                if trigger_time_local < datetime.now(local_tz):
                    logger.info(
                        f"Skipping game {game_pk} - trigger time is in the past"
                    )
                    skipped_count += 1
                    continue

                # Create Windows scheduled task
                task_name = f"BaseballPreGame_{game_pk}_{today_str.replace('-', '')}"

                # Format time for schtasks (HH:MM format)
                trigger_time_str = trigger_time_local.strftime("%H:%M")
                trigger_date_str = trigger_time_local.strftime("%m/%d/%Y")

                # Build schtasks command
                schtasks_cmd = [
                    "schtasks",
                    "/create",
                    "/tn",
                    task_name,
                    "/tr",
                    f'"{python_exe}" "{pre_game_script}" {game_pk}',
                    "/sc",
                    "once",
                    "/st",
                    trigger_time_str,
                    "/sd",
                    trigger_date_str,
                    "/f",  # Force overwrite if exists
                    "/rl",
                    "highest",  # Run with highest privileges
                ]

                # Execute the schtasks command
                try:
                    result = subprocess.run(
                        schtasks_cmd, capture_output=True, text=True, check=True
                    )
                    logger.info(
                        f"✅ Scheduled Windows task for game {game_pk} at {trigger_time_local}"
                    )
                    # Log schtasks output if there are any messages
                    if result.stdout.strip():
                        logger.debug(f"schtasks output: {result.stdout.strip()}")
                    scheduled_count += 1

                    # Also schedule task deletion (cleanup after 6 hours)
                    cleanup_time = trigger_time_local + timedelta(hours=6)
                    cleanup_task_name = f"Cleanup_{task_name}"
                    cleanup_time_str = cleanup_time.strftime("%H:%M")
                    cleanup_date_str = cleanup_time.strftime("%m/%d/%Y")

                    # Create a simple cleanup batch script
                    cleanup_script_path = script_dir / "cleanup_temp.bat"
                    cleanup_script_path.write_text(
                        f'schtasks /delete /tn "{task_name}" /f\n'
                        f'schtasks /delete /tn "{cleanup_task_name}" /f\n'
                        f'del "{cleanup_script_path}"\n'
                    )

                    cleanup_cmd = [
                        "schtasks",
                        "/create",
                        "/tn",
                        cleanup_task_name,
                        "/tr",
                        f'"{cleanup_script_path}"',
                        "/sc",
                        "once",
                        "/st",
                        cleanup_time_str,
                        "/sd",
                        cleanup_date_str,
                        "/f",
                    ]

                    cleanup_result = subprocess.run(
                        cleanup_cmd, capture_output=True, text=True
                    )
                    if cleanup_result.returncode == 0:
                        logger.debug(f"Scheduled cleanup for task {task_name}")
                    else:
                        logger.warning(
                            f"Failed to schedule cleanup task: {cleanup_result.stderr}"
                        )

                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to create scheduled task for game {game_pk}")
                    logger.error(f"Command: {' '.join(schtasks_cmd)}")
                    if e.stdout:
                        logger.error(f"stdout: {e.stdout}")
                    if e.stderr:
                        logger.error(f"stderr: {e.stderr}")
                    skipped_count += 1
                except Exception as e:
                    logger.error(
                        f"Error creating scheduled task for game {game_pk}: {e}"
                    )
                    skipped_count += 1

            except Exception as e:
                logger.error(f"Error processing game {game.get('game_id', 'N/A')}: {e}")
                skipped_count += 1

        logger.info(
            f"Successfully scheduled {scheduled_count} games, skipped {skipped_count}"
        )

    except Exception as e:
        logger.error(f"Error in Windows task scheduling: {e}", exc_info=True)
        return False

    return True


def run_daily_scheduling_apscheduler_windows():
    """
    Windows-compatible version that doesn't keep the process alive too long
    """
    logger.info("--- Starting Daily Scheduling (Windows Mode) ---")

    # Configure scheduler for shorter runs
    jobstores = {"default": SQLAlchemyJobStore(url="sqlite:///scheduler_jobs.sqlite")}
    executors = {"default": ThreadPoolExecutor(5)}
    job_defaults = {"coalesce": False, "max_instances": 2, "misfire_grace_time": 300}

    scheduler = BlockingScheduler(
        jobstores=jobstores,
        executors=executors,
        job_defaults=job_defaults,
        timezone=pytz.utc,
    )

    def run_pre_game_simulation(game_pk):
        """Execute pre-game simulation"""
        try:
            logger.info(f"Starting pre-game simulation for game {game_pk}")
            script_dir = Path(__file__).parent.absolute()
            simulation_script = script_dir / "main_pre_game_trigger.py"

            result = subprocess.run(
                [sys.executable, simulation_script, str(game_pk)],
                capture_output=True,
                text=True,
                cwd=script_dir,
                timeout=1800,
            )

            if result.returncode == 0:
                logger.info(f"Pre-game simulation completed for game {game_pk}")
                if result.stdout.strip():
                    logger.debug(f"Simulation output: {result.stdout.strip()}")
            else:
                logger.error(f"Pre-game simulation failed for game {game_pk}")
                if result.stderr:
                    logger.error(f"Error details: {result.stderr}")
                if result.stdout:
                    logger.error(f"Output: {result.stdout}")

        except subprocess.TimeoutExpired:
            logger.error(
                f"Pre-game simulation timed out for game {game_pk} (30 minute limit)"
            )
        except Exception as e:
            logger.error(f"Error in pre-game simulation for game {game_pk}: {e}")

    try:
        today = date.today()
        today_str = today.strftime("%Y-%m-%d")
        schedule = statsapi.schedule(date=today_str, sportId=1)

        if not schedule:
            logger.info("No games scheduled for today.")
            return True

        scheduled_count = 0

        for game in schedule:
            game_pk = game.get("game_id")
            game_start_str = game.get("game_datetime")
            status = game.get("status", "Unknown").lower()

            if not game_pk or not game_start_str:
                continue
            if any(
                word in status
                for word in ["tbd", "postponed", "cancelled", "suspended"]
            ):
                continue

            game_start_utc = datetime.fromisoformat(game_start_str)
            if game_start_utc.tzinfo is None:
                game_start_utc = pytz.utc.localize(game_start_utc)
            trigger_time_utc = game_start_utc - timedelta(minutes=55)

            if trigger_time_utc < datetime.now(pytz.utc):
                continue

            # Only schedule games for the next 8 hours to avoid long-running process
            hours_until_game = (
                trigger_time_utc - datetime.now(pytz.utc)
            ).total_seconds() / 3600
            if hours_until_game <= 8:
                job_id = f"pregame_{game_pk}"
                scheduler.add_job(
                    func=run_pre_game_simulation,
                    trigger="date",
                    run_date=trigger_time_utc,
                    args=[game_pk],
                    id=job_id,
                    replace_existing=True,
                )
                scheduled_count += 1
                logger.info(f"Scheduled game {game_pk} for {trigger_time_utc} UTC")

        if scheduled_count > 0:
            logger.info(
                f"Starting scheduler for {scheduled_count} games in the next 8 hours"
            )
            # Run scheduler but only for a limited time
            try:
                scheduler.start()
            except KeyboardInterrupt:
                logger.info("Scheduler interrupted")
        else:
            logger.info("No games to schedule in the next 8 hours")

    except Exception as e:
        logger.error(f"Error in APScheduler: {e}", exc_info=True)

    return True


# Main execution
if __name__ == "__main__":
    logger.info(f"Starting daily process at {datetime.now(tz=pytz.utc)}...")

    # Choose your preferred method:

    # Method 1: Update data then use Windows scheduled tasks (RECOMMENDED for Windows)
    update_success = run_incremental_update_and_feature_recalc()
    if update_success:
        create_windows_scheduled_tasks_for_games()  # Use this for Windows

    # Method 2: Update data then use APScheduler for near-term games only
    # update_success = run_incremental_update_and_feature_recalc()
    # if update_success:
    #     run_daily_scheduling_apscheduler_windows()  # Alternative approach

    logger.info(f"Daily process finished at {datetime.now(tz=pytz.utc)}.")
