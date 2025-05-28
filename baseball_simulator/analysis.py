import collections
import logging

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


# --- Main Analysis Function ---


def calculate_probabilities_and_odds(
    all_results: list,
    num_simulations: int,
    max_bin_val: int = 5,
):
    """
    Analyzes simulation results to calculate probabilities and odds for
    Hits (H), Walks (BB), and Home Runs (HR) per team per inning (0-5+).

    Args:
        all_results (list): List of dictionaries, each representing one simulation run.
                            Example element: {'inning_1': {'away': {'H':h,'R':r,'BB':bb,'HR':hr}, ...}}
        num_simulations (int): The total number of simulations run.
        max_bin_val (int): The threshold for the final '+' bin (e.g., 5 means bins 0,1,2,3,4,5+).

    Returns:
        polars.DataFrame: A DataFrame with columns: inning, team, stat, number_bin,
                          probability, decimal_odds, american_odds.
                          Returns None if num_simulations is zero or input is empty.

    """
    if not all_results or num_simulations <= 0:
        logger.warning(
            "No simulation results provided or num_simulations is zero. Cannot calculate probabilities.",
        )
        return None

    plus_bin_label = f"{max_bin_val}+"  # e.g., "5+"

    # --- Initialize dictionaries to store counts ---
    results_counts = {}
    for inn in [1, 2, 3]:
        results_counts[f"inning_{inn}"] = {
            "away": {
                "H": collections.defaultdict(int),
                "BB": collections.defaultdict(int),
                "HR": collections.defaultdict(int),
            },
            "home": {
                "H": collections.defaultdict(int),
                "BB": collections.defaultdict(int),
                "HR": collections.defaultdict(int),
            },
        }

    # --- Tally results into bins ---
    for sim_result in all_results:
        for inn in [1, 2, 3]:
            inn_key = f"inning_{inn}"
            if inn_key not in sim_result:
                continue  # Skip if inning missing

            for team in ["away", "home"]:
                if team not in sim_result[inn_key]:
                    continue  # Skip if team missing

                try:
                    team_result = sim_result[inn_key][team]
                    h = team_result.get("H", 0)  # Use .get() for safety
                    bb = team_result.get("BB", 0)
                    hr = team_result.get("HR", 0)

                    # Determine bin (0 to max_bin_val)
                    h_bin = min(h, max_bin_val)
                    bb_bin = min(bb, max_bin_val)
                    hr_bin = min(hr, max_bin_val)

                    # Increment count for the corresponding bin
                    results_counts[inn_key][team]["H"][h_bin] += 1
                    results_counts[inn_key][team]["BB"][bb_bin] += 1
                    results_counts[inn_key][team]["HR"][hr_bin] += 1
                except TypeError:
                    # Handle cases where H/BB/HR might not be numbers if sim failed partially
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
                    continue  # Skip this entry if structure is wrong

    # --- Calculate probabilities and odds, structure for DataFrame ---
    data_for_df = []
    for inn in [1, 2, 3]:
        inn_key = f"inning_{inn}"
        if inn_key not in results_counts:
            continue

        for team in ["away", "home"]:
            if team not in results_counts[inn_key]:
                continue

            for stat in ["H", "BB", "HR"]:
                if stat not in results_counts[inn_key][team]:
                    continue

                stat_counts = results_counts[inn_key][team][stat]

                # Calculate probabilities for bins 0 to max_bin_val
                for i in range(max_bin_val + 1):  # Iterate 0, 1, 2, 3, 4, 5
                    count = stat_counts.get(i, 0)
                    probability = count / num_simulations
                    decimal_odds = probability_to_decimal_odds(probability)
                    american_odds = probability_to_american_odds(probability)

                    # Determine bin label ('0', '1', ..., '4', '5+')
                    number_bin_label = f"{i}" if i < max_bin_val else plus_bin_label

                    data_for_df.append(
                        {
                            "inning": inn,
                            "team": team,
                            "stat": stat,
                            "number_bin": number_bin_label,
                            "probability": probability,
                            "decimal_odds": decimal_odds,
                            "american_odds": american_odds,
                        },
                    )
                    # Break after processing the max_bin_val bin (which represents X+)
                    if i == max_bin_val:
                        break

    # --- Create Polars DataFrame ---
    if not data_for_df:
        logger.warning("No data processed for DataFrame creation.")
        return pl.DataFrame()  # Return empty DataFrame

    schema = {
        "inning": pl.Int64,
        "team": pl.Categorical,
        "stat": pl.Categorical,
        "number_bin": pl.Utf8,  # e.g., '0', '1', '5+'
        "probability": pl.Float64,
        "decimal_odds": pl.Float64,  # Can be null
        "american_odds": pl.Int64,  # Can be null
    }

    df_probabilities = pl.DataFrame(data_for_df, schema=schema)

    return df_probabilities.sort(["inning", "team", "stat", "number_bin"])
