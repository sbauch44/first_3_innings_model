# Advanced Baseball Game Simulator

This project is a sophisticated baseball simulator designed to predict outcomes for the initial innings of Major League Baseball (MLB) games. It leverages a Bayesian statistical model, detailed player performance data, and various contextual factors (like park effects and defensive ratings) to generate probabilities for key game events such as hits, runs, walks, and home runs.

## Project Purpose

The primary goal of this project is to provide a data-driven approach to forecasting baseball game events, specifically focusing on the first three innings. The simulator aims to:

*   Generate granular probabilities for various plate appearance outcomes.
*   Offer insights into expected offensive performance for teams based on lineups, starting pitchers, park factors, and defensive strengths.
*   Serve as a foundational tool for further analysis, such as developing betting strategies, fantasy sports projections, or understanding game dynamics.

## Key Features

*   **Comprehensive Data Integration:**
    *   Fetches daily Statcast data for all MLB games.
    *   Retrieves and incorporates park factor information.
    *   Gathers team-level defensive statistics.
    *   Downloads Fangraphs player projections (batting and pitching).
*   **Advanced Statistical Modeling:**
    *   Calculates rolling, ballasted statistics for batters and pitchers to reflect current performance levels.
    *   Utilizes a pre-trained Bayesian hierarchical model to predict various plate appearance outcomes (e.g., Strikeout, Walk, Single, Double, Triple, HomeRun, Out_In_Play, HBP).
*   **Detailed Game Simulation:**
    *   Simulates the first three innings of specified MLB games.
    *   Considers batter vs. pitcher matchups, platoon advantages, park factors, and team defense.
    *   Runs thousands of simulations per game to generate robust probability distributions.
*   **Probabilistic Forecasting:**
    *   Outputs probabilities for the number of Hits (H), Runs (R), Walks (BB), and Home Runs (HR) for each team in each of the first three innings (binned as 0, 1, 2, 3, 4, 5+).
    *   Calculates corresponding decimal and American odds for these events.
*   **Automated Triggers (Conceptual):**
    *   Includes scripts designed for daily data updates (`main_daily_trigger.py`).
    *   Includes scripts for initiating pre-game simulations (`main_pre_game_trigger.py`), intended to be scheduled.

## Project Structure

The project is organized as follows:

*   `baseball_simulator/`: Contains the core Python scripts for data processing, simulation, and analysis.
    *   `main_daily_trigger.py`: Orchestrates the daily fetching and processing of data, recalculates player statistics, and (conceptually) schedules pre-game simulations.
    *   `main_pre_game_trigger.py`: Manages the simulation process for a specific game, including loading data, running simulations, and saving results.
    *   `simulator.py`: Defines the `BaseballSimulator` class, which handles the logic for simulating plate appearances, innings, and games using the Bayesian model.
    *   `data_fetcher.py`: Contains functions to retrieve data from various external sources (Statcast, Fangraphs, etc.).
    *   `data_processor.py`: Includes functions for cleaning, transforming, and preparing data for modeling and simulation, including the calculation of rolling statistics.
    *   `analysis.py`: Provides functions to analyze the raw simulation output and calculate probabilities and odds for game events.
    *   `model_loader.py`: Responsible for loading the pre-trained Bayesian model (`multi_outcome_model.nc`) and the associated feature scaler (`pa_outcome_scaler.joblib`).
    *   `storage.py`: Helper functions for saving and loading data, typically Parquet files.
    *   `config.py`: Stores configuration variables such as file paths, model parameters, league average rates, and other settings.
    *   `multi_outcome_model.nc`: The pre-trained Bayesian hierarchical model file (likely using NetCDF format, common with ArviZ/PyMC).
    *   `pa_outcome_scaler.joblib`: The saved scikit-learn scaler used for preprocessing features before feeding them into the model.
    *   `requirements.txt`: Lists the Python dependencies for the project.
*   `clean_data/`: Serves as the storage location for processed data files (in Parquet format). This includes historical plate appearance data, daily player stats, park factors, defensive stats, and Fangraphs projections.
    *   `results/`: A subdirectory within `clean_data` where simulation probabilities for individual games are stored, typically organized by date and game ID.
*   `.gitignore`: Specifies intentionally untracked files that Git should ignore.
*   `README.md`: This file, providing an overview of the project.

## Workflow

The project operates through two primary workflows:

1.  **Daily Data Update and Feature Recalculation:**
    *   This process is managed by `main_daily_trigger.py` and is intended to run once per day.
    *   **Data Fetching:** Gathers the latest data from the previous day (Statcast), along with updated park factors, defensive stats, and player projections from Fangraphs.
    *   **Data Integration:** The new Statcast data is merged with historical plate appearance records.
    *   **Feature Engineering:** Rolling statistics (e.g., K%, BB%, ISO for batters; K%/9, BB%/9 for pitchers) are recalculated for all players based on the updated dataset. These stats are "ballasted" using league averages to provide more stable estimates, especially for players with limited recent activity.
    *   **Data Storage:** All processed data (updated historical PAs, daily stats, park factors, defensive stats, projections) are saved to the `clean_data/` directory as Parquet files.
    *   **Scheduling (Conceptual):** Identifies games scheduled for the current day and (hypothetically) schedules the `main_pre_game_trigger.py` script to run before each game.

2.  **Pre-Game Simulation:**
    *   This process is managed by `main_pre_game_trigger.py` and is run for each individual game, ideally triggered automatically before the game starts.
    *   **Load Essential Data:**
        *   Loads the trained Bayesian model (`multi_outcome_model.nc`) and feature scaler (`pa_outcome_scaler.joblib`) via `model_loader.py`.
        *   Fetches game-specific information, including team IDs, venue, and date.
        *   Retrieves expected lineups and starting pitchers for the specific game (`data_fetcher.get_batting_orders`).
        *   Loads the relevant park factors and team defensive ratings from the files prepared by the daily process.
    *   **Prepare Simulation Inputs:** Using all the loaded data, `data_processor.prepare_simulation_inputs` compiles the feature sets required by the model for each batter and pitcher involved in the matchup. This includes their current rolling stats, opponent pitcher/batter stats, park effects, defensive context, and platoon advantages.
    *   **Run Simulations:** The `BaseballSimulator` class takes these inputs and:
        *   Predicts probabilities for various plate appearance outcomes (Single, Double, HR, Walk, Strikeout, etc.) for each potential matchup using the mean posterior parameters from the Bayesian model.
        *   Simulates the first three innings of the game thousands of times (configurable via `config.NUM_SIMULATIONS`). Each simulation plays through the innings batter by batter, updating the game state (outs, runners on base, score) based on the probabilistic outcomes of each plate appearance.
    *   **Analyze and Store Results:**
        *   The results from all simulations are aggregated by `analysis.calculate_probabilities_and_odds`.
        *   This function calculates the probability distribution for the number of Hits, Runs, Walks, and Home Runs for each team in each of the first three innings.
        *   The resulting probabilities and corresponding odds are saved to a Parquet file in the `clean_data/results/{game_date}/` directory.

## Technologies and Libraries

This project is built using Python and leverages several powerful libraries for data manipulation, statistical modeling, and API interaction:

*   **Python 3.x**
*   **Core Data Science & Numerics:**
    *   [Polars](https://pola.rs/): Used for high-performance data manipulation and DataFrame operations, especially with large datasets.
    *   [Pandas](https://pandas.pydata.org/): Utilized for data handling and analysis, particularly within the simulation components.
    *   [NumPy](https://numpy.org/): Essential for numerical computations, array operations, and supporting other scientific libraries.
*   **Statistical Modeling & Machine Learning:**
    *   [PyMC](https://www.pymc.io/welcome.html) / [ArviZ](https://arviz-devs.github.io/arviz/): The simulation model (`multi_outcome_model.nc`) is a Bayesian model, likely built using PyMC. ArviZ is commonly used with PyMC for processing and storing inference data (`idata`).
    *   [Scikit-learn](https://scikit-learn.org/): Used for machine learning tasks, including the feature scaling (`pa_outcome_scaler.joblib` which is a `StandardScaler`).
    *   [Joblib](https://joblib.readthedocs.io/): For saving and loading Python objects, particularly the Scikit-learn scaler.
*   **API Interaction & Data Fetching:**
    *   [StatsAPI](https://github.com/toddrob99/statsapi): Used to fetch MLB game schedules and other game-related information.
    *   (Implied) `requests` or similar for fetching data from sources like Fangraphs and Statcast (though not explicitly listed as a top-level dependency if wrapped in `data_fetcher`).
*   **Utilities:**
    *   `logging`: For tracking script execution and debugging.
    *   `pytz`: For timezone handling.

## Getting Started

To get the project up and running, follow these general steps:

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Set up a Python Environment:**
    It's recommended to use a virtual environment. Python 3.9 or newer is advisable.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    The project's dependencies are listed in `requirements.txt`.
    ```bash
    pip install -r baseball_simulator/requirements.txt
    ```
    *Note: The `requirements.txt` is located within the `baseball_simulator` directory.*

4.  **Initial Data Population:**
    *   Before running simulations, you need to populate the `clean_data/` directory. This typically involves:
        *   Acquiring historical Statcast data (the project seems to expect this to be incrementally built, but an initial historical backfill process might be needed if starting from scratch - this aspect is not fully detailed in the current scripts).
        *   Running the `main_daily_trigger.py` script at least once to generate initial processed files (like `daily_stats_final.parquet`, `park_factors.parquet`, etc.). You might need to adjust date ranges or configurations for an initial bulk load.
    *   Ensure the pre-trained model (`multi_outcome_model.nc`) and scaler (`pa_outcome_scaler.joblib`) are present in the `baseball_simulator/` directory.

5.  **Configuration:**
    *   Review `baseball_simulator/config.py`. You may need to adjust `BASE_FILE_PATH` if your `clean_data` directory is not located at the default relative path from the scripts. Other parameters like date ranges or simulation numbers might also be configured here.

6.  **Running the Simulator:**
    *   Once the data is set up, you can run the daily update process or specific pre-game simulations (see "How to Run" section).

## How to Run

Ensure you have completed the steps in the "Getting Started" section, particularly installing dependencies and setting up initial data.

### Daily Data Update and Feature Recalculation

To run the daily process that fetches new data, updates statistics, and prepares for the day's simulations:

```bash
python baseball_simulator/main_daily_trigger.py
```
*   This script is designed to be run once daily.
*   It will attempt to fetch data for "yesterday" relative to the current date.
*   It also (conceptually) schedules the pre-game triggers. In a deployed environment, this script would be automated via a cron job or a cloud scheduling service (e.g., AWS EventBridge, Google Cloud Scheduler).

### Pre-Game Simulation

To run a simulation for a specific game:

```bash
python baseball_simulator/main_pre_game_trigger.py <game_pk>
```
*   Replace `<game_pk>` with the actual MLB Game ID (e.g., `777822`).
*   This script is intended to be run before a game starts.
*   It requires that the daily data update has already been run for the relevant period and that lineups are available.
*   Simulation results (probabilities and odds) will be saved in the `clean_data/results/{game_date}/` directory.

**Example:**

To simulate game with ID `777822`:
```bash
python baseball_simulator/main_pre_game_trigger.py 777822
```

**Logging:**
Both scripts use Python's `logging` module. Log messages, including progress and potential errors, will be printed to the console. The verbosity and format are configured within each script.
