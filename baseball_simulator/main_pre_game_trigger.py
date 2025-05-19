import data_fetcher
import data_processor
import model_loader
from simulator import BaseballSimulator
import storage
import analysis
import config # To get column names etc.
import sys

def run_pre_game_simulation(game_pk):
    print(f"Running pre-game simulation for game_pk: {game_pk}")


    # 1. Load model and scaler (do this once if possible)
    loader = model_loader.ModelLoader()
    loader.set_paths(
    base_dir=config.MODEL_PATH, 
    model_filename="updated_model.nc", 
    scaler_filename="updated_scaler.joblib"
    )
    idata, scaler = loader.load_all()

    # 2. Get Lineups/Pitchers
    # Need game_info containing home/away team IDs, venue etc. from schedule first
    # This might need to be passed in or fetched again based on game_pk
    game_info = data_fetcher.get_game_info(game_pk) # Assumes function exists
    lineup_data = data_fetcher.get_batting_orders(game_pk)
    if not lineup_data or not lineup_data.get('home') or not lineup_data.get('away'):
        print(f"Could not get lineups for {game_pk}. Aborting.")
        return

    # 3. Prepare All Inputs
    if game_info is not None:
        sim_inputs = data_processor.prepare_simulation_inputs(
            game_info, lineup_data['home'], lineup_data['away'],
            lineup_data['home_pitcher_id'], lineup_data['away_pitcher_id']
        )
    # sim_inputs should contain home_lineup_with_stats, away_lineup_with_stats, etc.

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
        print(f"Starting {num_sims} simulations...")
        for _ in range(num_sims):
            run_result = simulator.simulate_first_three_innings(
                home_lineup=sim_inputs['home_lineup_with_stats'],
                away_lineup=sim_inputs['away_lineup_with_stats'],
                home_pitcher_inputs=sim_inputs['home_pitcher_inputs'],
                away_pitcher_inputs=sim_inputs['away_pitcher_inputs'],
                game_context=sim_inputs['game_context'],
            )
            all_runs_results.append(run_result)
        print("Simulations complete.")

        # 5. Analyze Results to get Probability DataFrame
        results_df = analysis.calculate_probabilities_and_odds(all_runs_results, num_sims) # Assumes function exists

        # 6. Store Results
        today_str = game_info['game_date'] # Get date from game info
        if results_df is not None:
            storage.save_simulation_results(results_df, today_str, game_pk)
            print(f"Results saved for game {game_pk}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        game_pk_arg = int(sys.argv[1]) # Get game_pk from command line argument/event payload
        run_pre_game_simulation(game_pk_arg)
    else:
        print("Error: Missing game_pk argument.")