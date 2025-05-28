import collections
import logging
from typing import Optional

import polars as pl

# --- Helper Functions for Odds Conversion ---

logger = logging.getLogger(__name__)


def probability_to_decimal_odds(probability: float) -> float | None:
    """Converts probability to decimal odds."""
    if probability is None or probability <= 0:
        return None  # Or np.inf, or a very large number like 9999
    # Avoid division by very small number leading to massive odds if desired
    # if probability < 1e-9: return 9999.0
    try:
        return 1.0 / probability
    except ZeroDivisionError:
        return None  # Or np.inf


def probability_to_american_odds(probability: float) -> int | None:
    """Converts probability to American odds."""
    if probability is None or probability <= 0 or probability >= 1:
        # Handle edge cases: P=0 -> infinite odds, P=1 -> infinitely favored
        # Representing these perfectly is hard, return None or placeholder
        if probability == 1:
            return -99999  # Example for P=1
        return None  # For P=0 or invalid

    if probability < 1e-9:  # noqa: PLR2004
        return 99999  # Avoid massive odds for tiny probs

    if probability < 0.5:  # noqa: PLR2004
        # Underdog calculation: +odds
        odds = (100 / probability) - 100
    else:
        # Favorite calculation: -odds
        odds = -1 * (probability / (1 - probability)) * 100

    # Round to nearest integer
    return round(odds)


def extract_team_names(game_info: dict) -> dict:
    """
    Extract team names from game_info dictionary.

    Args:
        game_info: Dictionary containing game information

    Returns:
        Dictionary with 'away' and 'home' team names

    """
    team_names = {}
    if game_info:
        team_names["away"] = game_info.get("away_name", "Away Team")
        team_names["home"] = game_info.get("home_name", "Home Team")
    else:
        team_names["away"] = "Away Team"
        team_names["home"] = "Home Team"
    return team_names


def initialize_results_counts(stats: list) -> dict:
    """
    Initialize the results counting structure.

    Args:
        stats: List of statistics to track (default: ["H", "BB", "HR", "R"])

    Returns:
        Nested dictionary structure for counting results

    """
    if stats is None:
        stats = ["H", "BB", "HR", "R"]

    results_counts = {}
    for inn in [1, 2, 3]:
        results_counts[f"inning_{inn}"] = {
            "away": {stat: collections.defaultdict(int) for stat in stats},
            "home": {stat: collections.defaultdict(int) for stat in stats},
        }
    return results_counts


def tally_simulation_results(
    all_results: list, results_counts: dict, max_bin_val: int, stats: list
) -> None:
    """
    Tally simulation results into bins.

    Args:
        all_results: List of simulation results
        results_counts: Dictionary to store counts (modified in place)
        max_bin_val: Maximum bin value before grouping into '+'
        stats: List of statistics to track

    """
    if stats is None:
        stats = ["H", "BB", "HR", "R"]

    for sim_result in all_results:
        for inn in [1, 2, 3]:
            inn_key = f"inning_{inn}"
            if inn_key not in sim_result:
                continue

            for team in ["away", "home"]:
                if team not in sim_result[inn_key]:
                    continue

                try:
                    team_result = sim_result[inn_key][team]

                    # Extract all stat values
                    stat_values = {}
                    for stat in stats:
                        stat_values[stat] = team_result.get(stat, 0)

                    # Bin all stats
                    for stat in stats:
                        stat_bin = min(stat_values[stat], max_bin_val)
                        results_counts[inn_key][team][stat][stat_bin] += 1

                except TypeError:
                    logger.warning(
                        "Invalid data type found for %s/%s in a sim result. Skipping entry.",
                        inn_key,
                        team,
                    )
                    continue
                except KeyError as e:
                    logger.warning(
                        "Missing key %s in %s/%s. Skipping entry.",
                        e,
                        inn_key,
                        team,
                    )
                    continue


def calculate_team_probabilities(
    results_counts: dict,
    num_simulations: int,
    team_names: dict,
    max_bin_val: int,
    stats: list,
) -> list:
    """
    Calculate probabilities for individual teams.

    Args:
        results_counts: Dictionary with tallied results
        num_simulations: Total number of simulations
        team_names: Dictionary mapping team keys to names
        max_bin_val: Maximum bin value
        stats: List of statistics to process

    Returns:
        List of dictionaries for DataFrame creation

    """
    if stats is None:
        stats = ["H", "BB", "HR", "R"]

    plus_bin_label = f"{max_bin_val}+"
    data_for_df = []

    for inn in [1, 2, 3]:
        inn_key = f"inning_{inn}"
        if inn_key not in results_counts:
            continue

        for team in ["away", "home"]:
            if team not in results_counts[inn_key]:
                continue

            for stat in stats:
                if stat not in results_counts[inn_key][team]:
                    continue

                stat_counts = results_counts[inn_key][team][stat]

                # Calculate probabilities for bins 0 to max_bin_val
                for i in range(max_bin_val + 1):
                    count = stat_counts.get(i, 0)
                    probability = count / num_simulations
                    decimal_odds = probability_to_decimal_odds(probability)
                    american_odds = probability_to_american_odds(probability)

                    number_bin_label = f"{i}" if i < max_bin_val else plus_bin_label

                    data_for_df.append(
                        {
                            "inning": inn,
                            "team": team.capitalize(),
                            "team_name": team_names[team],
                            "stat": stat,
                            "number_bin": number_bin_label,
                            "probability": probability,
                            "decimal_odds": decimal_odds,
                            "american_odds": american_odds,
                        }
                    )

    return data_for_df


def calculate_total_probabilities(
    results_counts: dict, num_simulations: int, max_bin_val: int, stats: list
) -> list:
    """
    Calculate combined probabilities for home + away totals.

    Args:
        results_counts: Dictionary with tallied results
        num_simulations: Total number of simulations
        max_bin_val: Maximum bin value
        stats: List of statistics to process

    Returns:
        List of dictionaries for DataFrame creation

    """
    if stats is None:
        stats = ["H", "BB", "HR", "R"]

    plus_bin_label = f"{max_bin_val}+"
    data_for_df = []

    for inn in [1, 2, 3]:
        inn_key = f"inning_{inn}"
        if inn_key not in results_counts:
            continue

        for stat in stats:
            # Combine home and away counts for each bin
            combined_counts = collections.defaultdict(int)

            # Add counts from both teams
            for team in ["away", "home"]:
                if (
                    team in results_counts[inn_key]
                    and stat in results_counts[inn_key][team]
                ):
                    for bin_val, count in results_counts[inn_key][team][stat].items():
                        combined_counts[bin_val] += count

            # Calculate probabilities for combined totals
            for i in range(max_bin_val + 1):
                count = combined_counts.get(i, 0)
                probability = count / num_simulations
                decimal_odds = probability_to_decimal_odds(probability)
                american_odds = probability_to_american_odds(probability)

                number_bin_label = f"{i}" if i < max_bin_val else plus_bin_label

                data_for_df.append(
                    {
                        "inning": inn,
                        "team": "Total",
                        "team_name": "Combined Total",
                        "stat": stat,
                        "number_bin": number_bin_label,
                        "probability": probability,
                        "decimal_odds": decimal_odds,
                        "american_odds": american_odds,
                    }
                )

    return data_for_df


def create_results_dataframe(data_for_df: list) -> pl.DataFrame:
    """
    Create a Polars DataFrame from the processed data.

    Args:
        data_for_df: List of dictionaries containing the processed results

    Returns:
        Polars DataFrame with results

    """
    if not data_for_df:
        logger.warning("No data processed for DataFrame creation.")
        return pl.DataFrame()

    schema = {
        "inning": pl.Int64,
        "team": pl.Categorical,
        "team_name": pl.Utf8,
        "stat": pl.Categorical,
        "number_bin": pl.Utf8,
        "probability": pl.Float64,
        "decimal_odds": pl.Float64,
        "american_odds": pl.Int64,
    }

    df_probabilities = pl.DataFrame(data_for_df, schema=schema)
    return df_probabilities.sort(["inning", "team", "stat", "number_bin"])


# --- Main Analysis Function (Now Modularized) ---
def calculate_probabilities_and_odds(
    all_results: list,
    num_simulations: int,
    game_info: dict,
    stats: Optional[list] = None,
    max_bin_val: int = 5,
    include_totals: bool = True,
):
    """
    Analyzes simulation results to calculate probabilities and odds for
    specified statistics per team per inning (0-5+).

    Args:
        all_results (list): List of dictionaries, each representing one simulation run.
                            Example element: {'inning_1': {'away': {'H':h,'R':r,'BB':bb,'HR':hr}, ...}}
        num_simulations (int): The total number of simulations run.
        game_info (dict): Dictionary containing game information including team names.
                         Expected keys: 'away_name', 'home_name'
        max_bin_val (int): The threshold for the final '+' bin (e.g., 5 means bins 0,1,2,3,4,5+).
        stats (list, optional): List of statistics to analyze. Default: ["H", "BB", "HR", "R"]
        include_totals (bool): Whether to include combined totals for home + away

    Returns:
        polars.DataFrame: A DataFrame with columns: inning, team, team_name, stat, number_bin,
                          probability, decimal_odds, american_odds.
                          Returns None if num_simulations is zero or input is empty.

    """
    if not all_results or num_simulations <= 0:
        logger.warning(
            "No simulation results provided or num_simulations is zero. Cannot calculate probabilities."
        )
        return None

    # Set default stats if not provided
    if stats is None:
        stats = ["H", "BB", "HR", "R"]

    # Step 1: Extract team names
    team_names = extract_team_names(game_info)

    # Step 2: Initialize counting structure
    results_counts = initialize_results_counts(stats)

    # Step 3: Tally simulation results
    tally_simulation_results(all_results, results_counts, max_bin_val, stats)

    # Step 4: Calculate team probabilities
    team_data = calculate_team_probabilities(
        results_counts, num_simulations, team_names, max_bin_val, stats
    )

    # Step 5: Calculate total probabilities (if requested)
    total_data = []
    if include_totals:
        total_data = calculate_total_probabilities(
            results_counts, num_simulations, max_bin_val, stats
        )

    # Step 6: Combine all data and create DataFrame
    all_data = team_data + total_data
    df_probabilities = create_results_dataframe(all_data)

    return df_probabilities
