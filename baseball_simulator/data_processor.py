import polars as pl
import pandas as pd
import numpy as np
import time
import datetime
from config import (
    RAW_COLS_TO_KEEP,
    OUTCOME_COL_NAME,
    K_EVENTS,
    BB_EVENTS,
    HBP_EVENTS,
    OUT_IN_PLAY_EVENTS,
    LEAGUE_AVG_RATES,
    BALLAST_WEIGHTS,
    END_YEAR,
)


# --- STATCAST FUNCTIONS ---
def process_statcast_data(df: pl.DataFrame,) -> pl.DataFrame:
    """
    Process the statcast data keep only relevant columns and plate outcomes.
    """
    df = (
        df
        .select(RAW_COLS_TO_KEEP)
    )

    # Sort data to ensure 'last()' picks the final pitch event
    df_pa = (
        df
        .sort(
            "game_pk", "at_bat_number", "pitch_number"
        )
        .group_by(
            "game_pk", "at_bat_number" # Group by unique PA identifier
        )
        .last() # Take the last pitch record for each PA
    )

    # --- Map Events to Categories using pl.when().then() ---
    df_with_outcome = (
        df_pa
        .with_columns(
            pl.when(pl.col("events") == "single").then(pl.lit(1))
            .when(pl.col("events") == "double").then(pl.lit(2))
            .when(pl.col("events") == "triple").then(pl.lit(3))
            .when(pl.col("events") == "home_run").then(pl.lit(4))
            .when(pl.col("events").is_in(K_EVENTS)).then(pl.lit(5))
            .when(pl.col("events").is_in(BB_EVENTS)).then(pl.lit(6))
            .when(pl.col("events").is_in(HBP_EVENTS)).then(pl.lit(7))
            .when(pl.col("events").is_in(OUT_IN_PLAY_EVENTS)).then(pl.lit(0))
            .otherwise(pl.lit(99)) # Assign 99 to nulls or any other unmapped event
            .alias(OUTCOME_COL_NAME)
        )
        .filter(pl.col(OUTCOME_COL_NAME) != 99)
    )

    return df_with_outcome


def create_helper_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create helper columns for analysis
    """
    df = (
        df
        .with_columns(
            is_pa = pl.col('events').is_not_null(),
            is_ab = pl.col('events').is_in(['single', 'double', 'triple', 'home_run',
                                            'strikeout', 'strikeout_double_play',
                                            'field_out', 'force_out', 'grounded_into_double_play',
                                            'double_play', 'triple_play', 'field_error',
                                            'fielders_choice_out', 'fielders_choice',
                                            ]),
            is_hit = pl.col('events').is_in(['single', 'double', 'triple', 'home_run']),
            is_k = pl.col('events').is_in(['strikeout', 'strikeout_double_play']),
            is_bb = pl.col('events').is_in(['walk', 'catcher_interf']),
            is_hbp = (pl.col('events') == 'hit_by_pitch'),
            is_1b = (pl.col('events') == 'single'),
            is_2b = (pl.col('events') == 'double'),
            is_3b = (pl.col('events') == 'triple'),
            is_hr = (pl.col('events') == 'home_run'),
            is_out = pl.col('events').is_in([
                'field_out', 'force_out', 'grounded_into_double_play',
                'double_play', 'triple_play', 'sac_fly', 'sac_bunt',
                'sac_fly_double_play', 'sac_bunt_double_play',
                'field_error', # Typically counts as an out for the model's purpose
                'fielders_choice_out',
                'fielders_choice',
            ]),
        )
    )
    return df


def calculate_batter_daily_totals(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate daily sums for batters.
    """
    df = (
        df
        .group_by("batter", "game_date")
        .agg(
            pl.sum("is_pa").alias("daily_pa"),
            pl.sum("is_ab").alias("daily_ab"),
            pl.sum("is_hit").alias("daily_h"),
            pl.sum("is_k").alias("daily_k"),
            pl.sum("is_bb").alias("daily_bb"),
            # Add sums for HBP, 1B, 2B, 3B, HR if needed for individual rate ballasts
            pl.sum("is_hbp").alias("daily_hbp"),
            pl.sum("is_1b").alias("daily_1b"),
            pl.sum("is_2b").alias("daily_2b"),
            pl.sum("is_3b").alias("daily_3b"),
            pl.sum("is_hr").alias("daily_hr"),
        )
        .sort("batter", "game_date")
    )
    return df


def calculate_pitcher_daily_totals(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate daily sums for pitchers.
    """
    df = (
        df
        .group_by("pitcher", "game_date")
        .agg(
            pl.sum("is_pa").alias("daily_pa"),
            pl.sum("is_ab").alias("daily_ab"),
            pl.sum("is_hit").alias("daily_h"),
            pl.sum("is_k").alias("daily_k"),
            pl.sum("is_bb").alias("daily_bb"),
            # Add sums for HBP, 1B, 2B, 3B, HR if needed for individual rate ballasts
            pl.sum("is_hbp").alias("daily_hbp"),
            pl.sum("is_1b").alias("daily_1b"),
            pl.sum("is_2b").alias("daily_2b"),
            pl.sum("is_3b").alias("daily_3b"),
            pl.sum("is_hr").alias("daily_hr"),
        )
        .sort("pitcher", "game_date")
    )
    return df


def calculate_cumulative_batter_stats(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate cumulative stats for batters.
    """
    df = (
        df
        .with_columns([
            pl.col("daily_pa").cum_sum().over("batter").sort_by("game_date").alias("tmp_cum_pa_prev_day"),
            pl.col("daily_ab").cum_sum().over("batter").sort_by("game_date").alias("tmp_cum_ab_prev_day"),
            pl.col("daily_h").cum_sum().over("batter").sort_by("game_date").alias("tmp_cum_h_prev_day"),
            pl.col("daily_k").cum_sum().over("batter").sort_by("game_date").alias("tmp_cum_k_prev_day"),
            pl.col("daily_bb").cum_sum().over("batter").sort_by("game_date").alias("tmp_cum_bb_prev_day"),
            pl.col("daily_hbp").cum_sum().over("batter").sort_by("game_date").alias("tmp_cum_hbp_prev_day"),
            pl.col("daily_1b").cum_sum().over("batter").sort_by("game_date").alias("tmp_cum_1b_prev_day"),
            pl.col("daily_2b").cum_sum().over("batter").sort_by("game_date").alias("tmp_cum_2b_prev_day"),
            pl.col("daily_3b").cum_sum().over("batter").sort_by("game_date").alias("tmp_cum_3b_prev_day"),
            pl.col("daily_hr").cum_sum().over("batter").sort_by("game_date").alias("tmp_cum_hr_prev_day"),
        ])
        .with_columns(
            pl.col("tmp_cum_pa_prev_day").shift(1).fill_null(0).alias("cum_pa_prev_day"),
            pl.col("tmp_cum_ab_prev_day").shift(1).fill_null(0).alias("cum_ab_prev_day"),
            pl.col("tmp_cum_h_prev_day").shift(1).fill_null(0).alias("cum_h_prev_day"),
            pl.col("tmp_cum_k_prev_day").shift(1).fill_null(0).alias("cum_k_prev_day"),
            pl.col("tmp_cum_bb_prev_day").shift(1).fill_null(0).alias("cum_bb_prev_day"),
            pl.col("tmp_cum_hbp_prev_day").shift(1).fill_null(0).alias("cum_hbp_prev_day"),
            pl.col("tmp_cum_1b_prev_day").shift(1).fill_null(0).alias("cum_1b_prev_day"),
            pl.col("tmp_cum_2b_prev_day").shift(1).fill_null(0).alias("cum_2b_prev_day"),
            pl.col("tmp_cum_3b_prev_day").shift(1).fill_null(0).alias("cum_3b_prev_day"),
            pl.col("tmp_cum_hr_prev_day").shift(1).fill_null(0).alias("cum_hr_prev_day"),
        )
        .drop(
            'tmp_cum_pa_prev_day', 'tmp_cum_ab_prev_day', 'tmp_cum_h_prev_day', 'tmp_cum_k_prev_day',
            'tmp_cum_bb_prev_day', 'tmp_cum_hbp_prev_day', 'tmp_cum_1b_prev_day', 'tmp_cum_2b_prev_day',
            'tmp_cum_3b_prev_day', 'tmp_cum_hr_prev_day',
        )
    )
    return df


def calculate_cumulative_pitcher_stats(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate cumulative stats for pitchers.
    """
    df = (
        df
        .with_columns([
            pl.col("daily_pa_a").cum_sum().over("pitcher").sort_by("game_date").alias("tmp_cum_pa_a_prev_day"),
            pl.col("daily_ab_a").cum_sum().over("pitcher").sort_by("game_date").alias("tmp_cum_ab_a_prev_day"),
            pl.col("daily_h_a").cum_sum().over("pitcher").sort_by("game_date").alias("tmp_cum_h_a_prev_day"),
            pl.col("daily_k_a").cum_sum().over("pitcher").sort_by("game_date").alias("tmp_cum_k_a_prev_day"),
            pl.col("daily_bb_a").cum_sum().over("pitcher").sort_by("game_date").alias("tmp_cum_bb_a_prev_day"),
            pl.col("daily_hbp_a").cum_sum().over("pitcher").sort_by("game_date").alias("tmp_cum_hbp_a_prev_day"),
            pl.col("daily_1b_a").cum_sum().over("pitcher").sort_by("game_date").alias("tmp_cum_1b_a_prev_day"),
            pl.col("daily_2b_a").cum_sum().over("pitcher").sort_by("game_date").alias("tmp_cum_2b_a_prev_day"),
            pl.col("daily_3b_a").cum_sum().over("pitcher").sort_by("game_date").alias("tmp_cum_3b_a_prev_day"),
            pl.col("daily_hr_a").cum_sum().over("pitcher").sort_by("game_date").alias("tmp_cum_hr_a_prev_day"),
        ])
        .with_columns(
            pl.col("tmp_cum_pa_a_prev_day").shift(1).fill_null(0).alias("cum_pa_a_prev_day"),
            pl.col("tmp_cum_ab_a_prev_day").shift(1).fill_null(0).alias("cum_ab_a_prev_day"),
            pl.col("tmp_cum_h_a_prev_day").shift(1).fill_null(0).alias("cum_h_a_prev_day"),
            pl.col("tmp_cum_k_a_prev_day").shift(1).fill_null(0).alias("cum_k_a_prev_day"),
            pl.col("tmp_cum_bb_a_prev_day").shift(1).fill_null(0).alias("cum_bb_a_prev_day"),
            pl.col("tmp_cum_hbp_a_prev_day").shift(1).fill_null(0).alias("cum_hbp_a_prev_day"),
            pl.col("tmp_cum_1b_a_prev_day").shift(1).fill_null(0).alias("cum_1b_a_prev_day"),
            pl.col("tmp_cum_2b_a_prev_day").shift(1).fill_null(0).alias("cum_2b_a_prev_day"),
            pl.col("tmp_cum_3b_a_prev_day").shift(1).fill_null(0).alias("cum_3b_a_prev_day"),
            pl.col("tmp_cum_hr_a_prev_day").shift(1).fill_null(0).alias("cum_hr_a_prev_day"),
        )
        .drop(
            'tmp_cum_pa_a_prev_day', 'tmp_cum_ab_a_prev_day', 'tmp_cum_h_a_prev_day', 'tmp_cum_k_a_prev_day',
            'tmp_cum_bb_a_prev_day', 'tmp_cum_hbp_a_prev_day', 'tmp_cum_1b_a_prev_day', 'tmp_cum_2b_a_prev_day',
            'tmp_cum_3b_a_prev_day', 'tmp_cum_hr_a_prev_day',
        )
    )
    return df


def calculate_ballasted_batter_stats(df: pl.DataFrame, lg_avgs=LEAGUE_AVG_RATES, ballast_weights=BALLAST_WEIGHTS) -> pl.DataFrame:
    """
    Calculate ballasted stats for batters.
    """
    df = (
        df
        .with_columns([
            (((pl.col("cum_h_prev_day") + lg_avgs['lg_avg'] * ballast_weights['batter']['is_hit']['value'])) /
            (pl.col("cum_ab_prev_day") + ballast_weights['batter']['is_hit']['value']))
            .alias("batter_avg_daily_input"),
            (((pl.col("cum_k_prev_day") + lg_avgs['lg_k_pct'] * ballast_weights['batter']['is_k']['value'])) /
            (pl.col("cum_pa_prev_day") + ballast_weights['batter']['is_k']['value']))
            .alias("batter_k_pct_daily_input"),
            (((pl.col("cum_bb_prev_day") + lg_avgs['lg_bb_pct']* ballast_weights['batter']['is_bb']['value']))/
            (pl.col("cum_pa_prev_day") + ballast_weights['batter']['is_bb']['value']))
            .alias("batter_bb_pct_daily_input"),
            (((pl.col("cum_hbp_prev_day") + lg_avgs['lg_hbp_pct'] * ballast_weights['batter']['is_hbp']['value'])) /
            (pl.col("cum_pa_prev_day") + ballast_weights['batter']['is_hbp']['value']))
            .alias("batter_hbp_pct_daily_input"),
            (((pl.col("cum_1b_prev_day") + lg_avgs['lg_1b_pct'] * ballast_weights['batter']['is_1b']['value'])) /
            (pl.col("cum_pa_prev_day") + ballast_weights['batter']['is_1b']['value']))
            .alias("batter_1b_pct_daily_input"),
            (((pl.col("cum_2b_prev_day") + lg_avgs['lg_2b_pct'] * ballast_weights['batter']['is_2b']['value'])) /
            (pl.col("cum_pa_prev_day") + ballast_weights['batter']['is_2b']['value']))
            .alias("batter_2b_pct_daily_input"),
            (((pl.col("cum_3b_prev_day") + lg_avgs['lg_3b_pct'] * ballast_weights['batter']['is_3b']['value'])) /
            (pl.col("cum_pa_prev_day") + ballast_weights['batter']['is_3b']['value']))
            .alias("batter_3b_pct_daily_input"),
            (((pl.col("cum_hr_prev_day") + lg_avgs['lg_hr_pct'] * ballast_weights['batter']['is_hr']['value'])) /
            (pl.col("cum_pa_prev_day") + ballast_weights['batter']['is_hr']['value']))
            .alias("batter_hr_pct_daily_input"),
        ])
        .with_columns(
            batter_non_k_out_pct_daily_input = ( 1 - (pl.sum_horizontal('batter_k_pct_daily_input', 'batter_bb_pct_daily_input', 'batter_hbp_pct_daily_input',
                                        'batter_1b_pct_daily_input', 'batter_2b_pct_daily_input', 'batter_3b_pct_daily_input', 'batter_hr_pct_daily_input'))
            )
        )
    )
    return df


def calculate_ballasted_pitcher_stats(df: pl.DataFrame, lg_avgs=LEAGUE_AVG_RATES, ballast_weights=BALLAST_WEIGHTS) -> pl.DataFrame:
    """
    Calculate ballasted stats for pitchers.
    """
    df = (
        df
        .with_columns([
            (((pl.col("cum_h_a_prev_day") + lg_avgs['lg_avg'] * ballast_weights['pitcher']['is_hit']['value'])) /
            (pl.col("cum_ab_a_prev_day") + ballast_weights['pitcher']['is_hit']['value']))
            .alias("pitcher_avg_a_daily_input"),
            (((pl.col("cum_k_a_prev_day") + lg_avgs['lg_k_pct'] * ballast_weights['pitcher']['is_k']['value'])) /
            (pl.col("cum_pa_a_prev_day") + ballast_weights['pitcher']['is_k']['value']))
            .alias("pitcher_k_pct_a_daily_input"),
            (((pl.col('cum_bb_a_prev_day')+ lg_avgs['lg_bb_pct'] * ballast_weights['pitcher']['is_bb']['value'])) /
            (pl.col('cum_pa_a_prev_day') + ballast_weights['pitcher']['is_bb']['value']))
            .alias("pitcher_bb_pct_a_daily_input"),
            (((pl.col("cum_hbp_a_prev_day") + lg_avgs['lg_hbp_pct'] * ballast_weights['pitcher']['is_hbp']['value'])) /
            (pl.col("cum_pa_a_prev_day") + ballast_weights['pitcher']['is_hbp']['value']))
            .alias("pitcher_hbp_pct_a_daily_input"),
            (((pl.col("cum_1b_a_prev_day") + lg_avgs['lg_1b_pct'] * ballast_weights['pitcher']['is_1b']['value'])) /
            (pl.col("cum_pa_a_prev_day") + ballast_weights['pitcher']['is_1b']['value']))
            .alias("pitcher_1b_pct_a_daily_input"),
            (((pl.col("cum_2b_a_prev_day") + lg_avgs['lg_2b_pct'] * ballast_weights['pitcher']['is_2b']['value'])) /
            (pl.col("cum_pa_a_prev_day") + ballast_weights['pitcher']['is_2b']['value']))
            .alias("pitcher_2b_pct_a_daily_input"),
            (((pl.col("cum_3b_a_prev_day") + lg_avgs['lg_3b_pct'] * ballast_weights['pitcher']['is_3b']['value'])) /
            (pl.col("cum_pa_a_prev_day") + ballast_weights['pitcher']['is_3b']['value']))
            .alias("pitcher_3b_pct_a_daily_input"),
            (((pl.col("cum_hr_a_prev_day") + lg_avgs['lg_hr_pct'] * ballast_weights['pitcher']['is_hr']['value'])) /
            (pl.col("cum_pa_a_prev_day") + ballast_weights['pitcher']['is_hr']['value']))
            .alias("pitcher_hr_pct_a_daily_input"),
        ])
        .with_columns(
            pitcher_non_k_out_pct_a_daily_input = ( 1 - (pl.sum_horizontal('pitcher_k_pct_a_daily_input', 'pitcher_bb_pct_a_daily_input', 'pitcher_hbp_pct_a_daily_input',
                                        'pitcher_1b_pct_a_daily_input', 'pitcher_2b_pct_a_daily_input', 'pitcher_3b_pct_a_daily_input', 'pitcher_hr_pct_a_daily_input'))
            )
        )
    )
    return df


def get_cols_to_join(df, position):
    """
    Get columns to join based on position.
    """
    if position not in ['batter', 'pitcher']:
        raise ValueError("Position must be 'batter' or 'pitcher'")

    cols = [col for col in df.columns if col.startswith(f"{position}_")]
    return cols


def select_subset_of_cols(df, position, cols):
    """
    Get columns to join based on position.
    """
    if position not in ['batter', 'pitcher']:
        raise ValueError("Position must be 'batter' or 'pitcher'")
    df = (
        df
        .select([
            position,
            "game_date",
            *cols,
        ])
    )
    return df


def join_together_final_df(main_df, df_bat, df_pitch):
    """
    Join the main DataFrame with batter and pitcher DataFrames.
    """
    main_df = (
        main_df
        .join(
            df_bat,
            on=["batter", "game_date"],
            how="left"
            )
        .join(
            df_pitch,
            on=["pitcher", "game_date"],
            how="left"
            )
        .with_columns(
            is_platoon_adv = (
                pl.when((pl.col('stand') == 'L') & (pl.col('p_throws') == 'R')).then(pl.lit(1))
                .when((pl.col('stand') == 'R') & (pl.col('p_throws') == 'L')).then(pl.lit(1))
            .otherwise(pl.lit(0))
            ),
            is_batter_home =  (pl.when(pl.col("inning_topbot") == "Bot") # Bottom of inning means home team batting
                .then(pl.lit(1))
                .otherwise(pl.lit(0)) # Top of inning means away team batting
                .cast(pl.Int8) # Cast to integer
            )
        )
    )
    return main_df


# DEFENSIVE STATS FUNCTIONS
def calculate_defensive_innings_played(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate defensive stats for each player from main statcast DataFrame.
    This function will melt the fielder columns into a long format and count unique game-inning-player combinations.
    """
    fielder_cols = [f"fielder_{i}" for i in range(2, 10)] # Fielder 2 (C) to Fielder 9 (RF)

    df_long = df.unpivot(
        index=["game_pk", "game_year", "inning"], # Columns to keep identifying the context
        on=fielder_cols, # Columns containing player IDs to be 'melted'
        variable_name="position_num_str", # New column for the original column name (e.g., 'fielder_2')
        value_name="player_id" # New column for the player ID value
    )

    df_player_innings = (
        df_long
        .filter(
            pl.col("player_id").is_not_null() # Ensure player_id is not missing
        )
        .unique(
            subset=["game_pk", "game_year", "inning", "player_id"]
        )
    )

    df_total_innings = (
        df_player_innings
        .group_by("player_id", "game_year")
        .agg(
            # Count the number of unique game-inning rows for each player-year combination
            pl.len().alias("total_innings_played")
        )
        .sort("player_id", "game_year")
    )
    return df_total_innings


def calculate_cumulative_defensive_stats(df: pl.DataFrame, df_total_innings: pl.DataFrame, end_year=END_YEAR) -> pl.DataFrame:
    """
    Calculate cumulative defensive stats for each player.
    """
    unique_players = df["player_id"].unique()

    df_all_players = pl.DataFrame({'player_id': unique_players})
    df_all_years = pl.DataFrame({'year': [x for x in range(2021, end_year + 1)]})

    # Cross join to get every player paired with every target year
    df_grid = df_all_players.join(df_all_years, how='cross').sort("player_id", "year")

    df_full_history = (
        df_grid
        .join(
            df, # Your original data
            on=["player_id", "year"],
            how="left"
        )
        .select(
            'player_id',
            'year',
            'outs_above_average',
        )
        .join(
            df_total_innings,
            left_on=['player_id', 'year'],
            right_on=['player_id', 'game_year'],
            how='left',
        )
        .with_columns([
            pl.col("outs_above_average").fill_null(0),
            pl.col("total_innings_played").fill_null(0),
        ])
        .sort("player_id", "year")
    )

    df_final_cumulative = (
        df_full_history
        .with_columns(
            pl.col("outs_above_average").cum_sum().over("player_id").alias("cumulative_oaa_temp"),
            pl.col("total_innings_played").cum_sum().over("player_id").alias("cumulative_innings_temp")
        )
        .with_columns(
            pl.col("cumulative_oaa_temp").shift(1).over("player_id").fill_null(0).alias("cumulative_oaa_prior"),
            pl.col("cumulative_innings_temp").shift(1).over("player_id").fill_null(0).alias("cumulative_innings_prior")
        )
        .drop("cumulative_oaa_temp", "cumulative_innings_temp") # Drop temporary columns
        .with_columns(
            outs_above_average_per_inning = (pl.col("cumulative_oaa_prior") / pl.col("cumulative_innings_prior")).fill_nan(0),
        )
    )

    return df_final_cumulative


# PARK FACTORS FUNCTIONS
def calculate_park_factors(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate park factors for each MLB ballpark.
    """
    clean_park_factors = (
        df
        .select(
            'venue_id',
            'venue_name',
            'main_team_id',
            'name_display_club',
            'metric_value_2025',
            'metric_value_2024',
            'metric_value_2023',
            'metric_value_2022',
            'metric_value_2021',
        )
        .unpivot(
            index=['venue_id', 'venue_name', 'main_team_id', 'name_display_club'],
            on=['metric_value_2025', 'metric_value_2024', 'metric_value_2023', 'metric_value_2022', 'metric_value_2021'],
            variable_name='year',
            value_name='park_factor'
        )
        .with_columns(
            year = pl.col('year').str.replace('metric_value_', '').cast(pl.Int64),
            park_factor = pl.col('park_factor').cast(pl.Float64),
        )
    )
    return clean_park_factors