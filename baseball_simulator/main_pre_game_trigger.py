import logging
import sys

import analysis
import config  # To get column names etc.
import data_fetcher
import data_processor
import model_loader
import storage
from simulator import BaseballSimulator

logger = logging.getLogger(__name__)


def run_pre_game_simulation(game_pk):
    """
    Runs a pre-game simulation for a given baseball game.

    This function performs the following steps:
        1. Loads the predictive model and scaler required for simulation.
        2. Fetches game information, including lineups and pitchers, for the specified game.
        3. Loads necessary data files such as park factors and defensive statistics.
        4. Prepares all simulation inputs using the fetched data.
        5. Runs the simulation for the first three innings multiple times (as specified in config).
        6. Analyzes the simulation results to calculate probabilities and odds.
        7. Stores the simulation results for later use.

    Args:
        game_pk (int or str): The unique identifier for the game to simulate.

    Returns:
        None. Results are saved via the storage module.
    Logs:
        - Information and error messages regarding the simulation process and any issues encountered.

    """
    logger.info("Running pre-game simulation for game_pk: %s", game_pk)

    # 1. Load model and scaler (do this once if possible)
    loader = model_loader.ModelLoader()
    loader.set_paths(
        base_dir=config.MODEL_PATH,
        model_filename="multi_outcome_model.nc",
        scaler_filename="pa_outcome_scaler.joblib",
    )
    idata, scaler = loader.load_all()

    # 2. Get Lineups/Pitchers
    # Need game_info containing home/away team IDs, venue etc. from schedule first
    # This might need to be passed in or fetched again based on game_pk
    game_info = data_fetcher.get_game_info(game_pk)
    if not game_info:
        logger.error("Could not get game info for %s. Aborting.", game_pk)
        return
    lineup_data = data_fetcher.get_batting_orders(game_pk)
    if not lineup_data or not lineup_data.get("home") or not lineup_data.get("away"):
        logger.error("Could not get lineups for %s. Aborting.", game_pk)
        return

    park_factors_df = storage.load_dataframe("park_factors.parquet")
    player_defense_df = storage.load_dataframe("defensive_stats.parquet")

    if park_factors_df is None or player_defense_df is None:
        logger.error("Required data files are missing. Aborting simulation.")
        return

    # 3. Prepare All Inputs
    sim_inputs = None
    if game_info is not None:
        today_str = game_info["game_date"]
        sim_inputs = data_processor.prepare_simulation_inputs(
            game_info=game_info,
            lineup_data=lineup_data,
            park_factors_df=park_factors_df,
            player_defense_df=player_defense_df,
        )
    # sim_inputs should contain home_lineup_with_stats, away_lineup_with_stats, etc.

    if sim_inputs is None:
        logger.error("Simulation inputs could not be prepared. Aborting simulation.")
        return
    # 4. Run Simulation (Many times)
    simulator = BaseballSimulator(
        idata=idata,
        scaler=scaler,
        outcome_labels=config.OUTCOME_LABELS,
        predictor_cols=config.PREDICTOR_COLS,
        continuous_cols=config.CONTINUOUS_COLS,
        categorical_cols=config.CATEGORICAL_COLS,
        league_avg_rates=config.LEAGUE_AVG_RATES,
    )

    num_sims = config.NUM_SIMULATIONS
    all_runs_results = []
    logger.info("Starting %d simulations...", num_sims)
    for _ in range(num_sims):
        run_result = simulator.simulate_first_three_innings(
            home_lineup=sim_inputs["home_lineup_with_stats"],
            away_lineup=sim_inputs["away_lineup_with_stats"],
            home_pitcher_inputs=sim_inputs["home_pitcher_inputs"],
            away_pitcher_inputs=sim_inputs["away_pitcher_inputs"],
            game_context=sim_inputs["game_context"],
        )
        all_runs_results.append(run_result)
    logger.info("Simulations complete.")

    # 5. Analyze Results to get Probability DataFrame
    results_df = analysis.calculate_probabilities_and_odds(
        all_runs_results,
        num_sims,
    )  # Assumes function exists

    # 6. Store Results
    if results_df is not None:
        storage.save_simulation_results(results_df, today_str, game_pk)  # type: ignore[attr-defined]
        logger.info("Results saved for game %s", game_pk)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        game_pk_arg = int(
            sys.argv[1],
        )  # Get game_pk from command line argument/event payload
        run_pre_game_simulation(game_pk_arg)
    else:
        logger.error("Error: Missing game_pk argument.")
