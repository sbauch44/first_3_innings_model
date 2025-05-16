import polars as pl
from pathlib import Path


NUM_SIMULATIONS = 10_000
 
# --- File Paths & Names ---
BASE_FILE_PATH = '/content/drive/My Drive/Betting Models/mlb/hits_model/' # Or use environment variables
RAW_STATCAST_FILES = [
    '2021_statcast_data.parquet',
    '2022_statcast_data.parquet',
    '2023_statcast_data.parquet',
    '2024_statcast_data.parquet',
]

HISTORICAL_PA_HELPERS_FILE = 'historical_pa_data_with_helpers.parquet'

CLEAN_STATCAST_FILE = 'clean_statcast_data.parquet'
BALLASTED_STATCAST_FILE = 'ballasted_statcast_data.parquet'
RAW_DEFENSE_FILE = 'statcast_defensive_stats.parquet'
CLEAN_DEFENSE_FILE = 'clean_defensive_stats.parquet'
RAW_PARKFACTOR_FILE = 'statcast_park_factors.parquet'
CLEAN_PARKFACTOR_FILE = 'clean_park_factors.parquet'
MODEL_PATH = "multi_outcome_model.nc" # Added example
SCALER_PATH = "pa_outcome_scaler.joblib" # Added example

# --- Data Selection ---
RAW_COLS_TO_KEEP = [
    "game_pk", "at_bat_number", "pitch_number", "batter", "pitcher",
    "events", "stand", "p_throws", "inning_topbot", "home_team", "away_team",
    "game_date", "game_type", "bb_type", "balls", "strikes", "outs_when_up",
    "inning", "game_year", "fielder_2", "fielder_3", "fielder_4", "fielder_5",
    "fielder_6", "fielder_7", "fielder_8", "fielder_9",
]

# --- Feature Engineering & Model ---
OUTCOME_COL_NAME = "pa_outcome_category"
START_YEAR = 2021
END_YEAR = 2025 # Or dynamically set
MODEL_TRAIN_YEARS = [2023, 2024]
LEAGUE_AVG_YEARS = [2021, 2022]

# Event Definitions (for helper columns and outcome mapping)
HIT_EVENTS = ['single', 'double', 'triple', 'home_run']
AB_EVENTS = [ # Events counting as an At Bat - REVIEW CAREFULLY!
    'single', 'double', 'triple', 'home_run', 'strikeout', 'strikeout_double_play',
    'field_out', 'force_out', 'grounded_into_double_play', 'double_play', 'triple_play',
    'field_error', 'fielders_choice', 'fielders_choice_out'
    # Excludes: 'walk', 'hit_by_pitch', 'sac_fly', 'sac_bunt', 'catcher_interf', 'intent_walk' ...
]
K_EVENTS = ['strikeout', 'strikeout_double_play']
BB_EVENTS = ['walk', 'catcher_interf'] # Include catcher interference? Check rules. Often rare.
HBP_EVENTS = ['hit_by_pitch']
OUT_IN_PLAY_EVENTS = [ # Used for outcome category 0
    'field_out', 'force_out', 'grounded_into_double_play',
    'double_play', 'triple_play', 'sac_fly', 'sac_bunt',
    'sac_fly_double_play', 'sac_bunt_double_play',
    'field_error', 'fielders_choice_out', 'fielders_choice'
]

# League Average Rates (for model input)
LEAGUE_AVG_RATES = {
    'lg_avg': 0.2430046187210952, 
    'lg_k_pct': 0.22924091208889638, 
    'lg_bb_pct': 0.08148381370630434, 
    'lg_hbp_pct': 0.011473038224461158, 
    'lg_1b_pct': 0.1402169753195324, 
    'lg_2b_pct': 0.0434629195855918, 
    'lg_3b_pct': 0.0035882083115064234, 
    'lg_hr_pct': 0.030792907181573653, 
    'lg_out_pct': 0.6889821376710302,
}

# Ballast Weights (Stabilization Points)
BALLAST_WEIGHTS = {
    'batter': {
        # Outcome: Corresponding Rate & Stabilization Point
        'is_hit': {'rate': 'AVG', 'value': 910, 'unit': 'AB'}, # Overall Hit Rate = AVG
        'is_k': {'rate': 'K%', 'value': 60, 'unit': 'PA'}, # Strikeout Rate
        'is_bb': {'rate': 'BB%', 'value': 120, 'unit': 'PA'}, # Walk Rate
        'is_hbp': {'rate': 'HBP%', 'value': 240, 'unit': 'PA'}, # Hit By Pitch Rate
        'is_1b': {'rate': '1B%', 'value': 290, 'unit': 'PA'}, # Single Rate
        'is_2b': {'rate': '2B%', 'value': 1600, 'unit': 'PA'}, # No specific stabilization point found for 2B Rate alone
        'is_3b': {'rate': '3B%', 'value': 1600, 'unit': 'PA'}, # No specific stabilization point found for 3B Rate alone
        'is_hr': {'rate': 'HR%', 'value': 170, 'unit': 'PA'}, # Home Run Rate
    },
    'pitcher': {
        'is_hit': {'rate': 'AVG_A', 'value': 630, 'unit': 'BF'}, # Overall Hit Rate Allowed = AVG Against (using BF)
        'is_k': {'rate': 'K%_A', 'value': 70, 'unit': 'BF'}, # Strikeout Rate Against
        'is_bb': {'rate': 'BB%_A', 'value': 170, 'unit': 'BF'}, # Walk Rate Against
        'is_hbp': {'rate': 'HBP%_A', 'value': 640, 'unit': 'BF'}, # Hit By Pitch Rate Against
        'is_1b': {'rate': '1B%_A', 'value': 670, 'unit': 'BF'}, # Single Rate Against
        'is_2b': {'rate': '2B%_A', 'value': 1450, 'unit': 'BF'}, # No specific stabilization point found
        'is_3b': {'rate': '3B%_A', 'value': 1450, 'unit': 'BF'}, # No specific stabilization point found
        'is_hr': {'rate': 'HR%_A', 'value': 1320, 'unit': 'BF'}, # Home Run Rate Against (Note: high stabilization)
    },
}

# Predictor Columns (Final list for model input)
# Define these based on the final rolling stats columns you create + context flags
PREDICTOR_COLS = [
    'is_platoon_adv',
    'is_batter_home',
    # Pitcher Stats
    # 'pitcher_avg_a_daily_input',
    'pitcher_k_pct_a_daily_input',
    'pitcher_bb_pct_a_daily_input',
    'pitcher_hbp_pct_a_daily_input',
    'pitcher_1b_pct_a_daily_input',
    'pitcher_2b_pct_a_daily_input',
    'pitcher_3b_pct_a_daily_input',
    'pitcher_hr_pct_a_daily_input',
    'pitcher_non_k_out_pct_a_daily_input',
    # Add other pitcher rate inputs here (HBP%, 1B%, 2B%, 3B%, HR%) if calculated
    # Batter Stats
    # 'batter_avg_daily_input',
    'batter_k_pct_daily_input',
    'batter_bb_pct_daily_input',
    'batter_hbp_pct_daily_input',
    'batter_1b_pct_daily_input',
    'batter_2b_pct_daily_input',
    'batter_3b_pct_daily_input',
    'batter_hr_pct_daily_input',
    'batter_non_k_out_pct_daily_input',
    # Add other batter rate inputs here (HBP%, 1B%, 2B%, 3B%, HR%) if calculated
    # Context Stats
    'team_defense_oaa_input',
    'park_factor_input',
]

CONTINUOUS_COLS = [
    # 'pitcher_avg_a_daily_input',
    'pitcher_k_pct_a_daily_input',
    'pitcher_bb_pct_a_daily_input',
    'pitcher_hbp_pct_a_daily_input',
    'pitcher_1b_pct_a_daily_input',
    'pitcher_2b_pct_a_daily_input',
    'pitcher_3b_pct_a_daily_input',
    'pitcher_hr_pct_a_daily_input',
    'pitcher_non_k_out_pct_a_daily_input',
    # 'batter_avg_daily_input',
    'batter_k_pct_daily_input',
    'batter_bb_pct_daily_input',
    'batter_hbp_pct_daily_input',
    'batter_1b_pct_daily_input',
    'batter_2b_pct_daily_input',
    'batter_3b_pct_daily_input',
    'batter_hr_pct_daily_input',
    'batter_non_k_out_pct_daily_input',
    'team_defense_oaa_input',
    'park_factor_input',
    # Add other continuous rate inputs here
]

CATEGORICAL_COLS = ['is_platoon_adv', 'is_batter_home']

# Outcome Mapping
OUTCOME_LABELS = { # Added example
    0: "Out_In_Play", 1: "Single", 2: "Double", 3: "Triple", 4: "HomeRun",
    5: "Strikeout", 6: "Walk", 7: "HBP"
}
N_CATEGORIES = len(OUTCOME_LABELS)

team_mapping = {
    'BOS': 'Red Sox',
    'MIN': 'Twins',
    'LAD': 'Dodgers',
    'CLE': 'Guardians',
    'SD': 'Padres',
    'KC': 'Royals',
    'CWS': 'White Sox',
    'WSH': 'Nationals',
    'TB': 'Rays',
    'AZ': 'D-backs',
    'MIL': 'Brewers',
    'STL': 'Cardinals',
    'DET': 'Tigers',
    'TEX': 'Rangers',
    'OAK': 'Athletics',
    'ATL': 'Braves',
    'PHI': 'Phillies',
    'CIN': 'Reds',
    'NYY': 'Yankees',
    'SEA': 'Mariners',
    'NYM': 'Mets',
    'LAA': 'Angels',
    'MIA': 'Marlins',
    'SF': 'Giants',
    'PIT': 'Pirates',
    'COL': 'Rockies',
    'HOU': 'Astros',
    'BAL': 'Orioles',
    'TOR': 'Blue Jays',
    'CHC': 'Cubs'
}

MAPPING_DF = (
    pl.DataFrame({
        "team_name": list(team_mapping.values()),
        "team_abbr": list(team_mapping.keys()),
    })
)