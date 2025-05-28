import json
import logging
import re

import config
import polars as pl
import requests
import statsapi
from pybaseball import (
    statcast,
    statcast_fielding,
)


def fetch_statcast_data():
    data = statcast(verbose=True)
    pl_data = pl.from_pandas(data)
    if pl_data.is_empty():
        msg = "No data found in the response."
        raise ValueError(msg)
    return pl_data


def fetch_park_factors():
    url = "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors?type=venue&year=2025&batSide=&stat=index_wOBA&condition=All&rolling=3&parks=mlb"

    response = requests.get(url, timeout=20)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code}")

    search_res = re.search(r"data = (.*);", response.text)
    data = search_res.group(1) if search_res is not None else None
    if data is not None:
        json_data = json.loads(data)
        df = pl.DataFrame(json_data)
        return df
    ValueError("No data found in the response.")
    return None


def fetch_defensive_stats(year):
    df_tmp = statcast_fielding.statcast_outs_above_average(year=year, pos="ALL")
    pdf = pl.from_pandas(df_tmp)
    if pdf.is_empty():
        raise ValueError("No data found for the specified year.")
    return pdf


def fetch_fangraphs_projections(
    position=None,
    cookies=config.FANGRAPHS_COOKIES,
    headers=config.FANGRAPHS_HEADERS,
):
    if position not in ["pit", "bat"]:
        raise ValueError("Position must be 'pit' or 'bat'")

    params = {
        "type": "steamerr",
        "stats": position,
        "pos": "all",
        "team": "0",
        "players": "0",
        "lg": "all",
        "z": "1747738554",
        "download": "1",
    }
    response = requests.get(
        "https://www.fangraphs.com/api/projections",
        params=params,
        cookies=cookies,
        headers=headers,
        timeout=20,
    )
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code}")
    df = pl.DataFrame(response.json())
    if df.is_empty():
        raise ValueError("No data found in the response.")
    return df


def get_batting_orders(game_pk):
    game_json = statsapi.get("game", {"gamePk": game_pk})
    boxscores = game_json["liveData"]["boxscore"]["teams"]
    home_batters = boxscores["home"]["battingOrder"]
    away_batters = boxscores["away"]["battingOrder"]
    home_pitchers = boxscores["home"]["pitchers"]
    away_pitchers = boxscores["away"]["pitchers"]

    lineup_data = {
        "home": {
            "batter_ids": home_batters,
            "pitcher_id": home_pitchers,
            "fielder_ids": home_batters,
        },
        "away": {
            "batter_ids": away_batters,
            "pitcher_id": away_pitchers,
            "fielder_ids": away_batters,
        },
    }

    return lineup_data


def get_game_info(game_pk):
    sched = statsapi.schedule()
    game_dict = next((game for game in sched if game["game_id"] == game_pk), None)
    return game_dict


def fetch_player_handedness(player_ids: list[int]) -> pl.DataFrame:
    """
    Fetch batting and pitching handedness for multiple players using statsapi.

    Args:
        player_ids: List of MLB player IDs to fetch handedness for

    Returns:
        Polars DataFrame with columns: player_id, bat_side, pitch_hand
        bat_side and pitch_hand will be 'L', 'R', or 'S' (switch)
        Returns None for players not found

    """
    if not player_ids:
        logging.warning("No player IDs provided to fetch_player_handedness")
        return pl.DataFrame(
            schema={
                "player_id": pl.Int64,
                "bat_side": pl.Utf8,
                "pitch_hand": pl.Utf8,
            },
        )

    # Convert to comma-separated string for API call
    player_ids_str = ",".join(str(pid) for pid in player_ids)

    try:
        logging.info(f"Fetching handedness data for {len(player_ids)} players...")

        # Make API call to get player info
        response = statsapi.get("people", {"personIds": player_ids_str})

        if not response or "people" not in response:
            logging.error("Invalid response from statsapi people endpoint")
            return pl.DataFrame(
                schema={
                    "player_id": pl.Int64,
                    "bat_side": pl.Utf8,
                    "pitch_hand": pl.Utf8,
                },
            )

        people_data = response["people"]
        logging.info(f"Received data for {len(people_data)} players")

        # Extract handedness data
        handedness_data = []

        for player in people_data:
            player_id = player.get("id")

            # Extract batting side
            bat_side_info = player.get("batSide", {})
            bat_side = bat_side_info.get("code", "R") if bat_side_info else "R"

            # Extract pitching hand
            pitch_hand_info = player.get("pitchHand", {})
            pitch_hand = pitch_hand_info.get("code", "R") if pitch_hand_info else "R"

            handedness_data.append(
                {
                    "player_id": player_id,
                    "bat_side": bat_side,
                    "pitch_hand": pitch_hand,
                },
            )

            logging.debug(f"Player {player_id}: bats {bat_side}, throws {pitch_hand}")

        # Create DataFrame
        df = pl.DataFrame(
            handedness_data,
            schema={
                "player_id": pl.Int64,
                "bat_side": pl.Utf8,
                "pitch_hand": pl.Utf8,
            },
        )

        # Check for missing players
        found_ids = set(df["player_id"].to_list())
        missing_ids = set(player_ids) - found_ids

        if missing_ids:
            logging.warning(f"No handedness data found for player IDs: {missing_ids}")

            # Add missing players with default values
            missing_data = []
            for missing_id in missing_ids:
                missing_data.append(
                    {
                        "player_id": missing_id,
                        "bat_side": "R",  # Default to right-handed
                        "pitch_hand": "R",
                    },
                )
                logging.warning(
                    f"Using default handedness (R/R) for player {missing_id}",
                )

            if missing_data:
                missing_df = pl.DataFrame(
                    missing_data,
                    schema={
                        "player_id": pl.Int64,
                        "bat_side": pl.Utf8,
                        "pitch_hand": pl.Utf8,
                    },
                )
                df = pl.concat([df, missing_df], how="vertical")

        logging.info(f"Successfully fetched handedness for {len(df)} players")
        return df.sort("player_id")

    except Exception as e:
        logging.error(f"Error fetching player handedness: {e}", exc_info=True)

        # Return DataFrame with defaults for all requested players
        default_data = []
        for player_id in player_ids:
            default_data.append(
                {
                    "player_id": player_id,
                    "bat_side": "R",
                    "pitch_hand": "R",
                },
            )

        return pl.DataFrame(
            default_data,
            schema={
                "player_id": pl.Int64,
                "bat_side": pl.Utf8,
                "pitch_hand": pl.Utf8,
            },
        )


def get_game_handedness_data(lineup_data: dict) -> pl.DataFrame:
    """
    Get handedness data for all players in a game's lineups.

    Args:
        lineup_data: Dictionary containing home/away lineups from get_batting_orders()

    Returns:
        Polars DataFrame with handedness data for all players in the game

    """
    # Collect all unique player IDs from the game
    all_player_ids = set()

    # Add batters
    if "home" in lineup_data and "batter_ids" in lineup_data["home"]:
        all_player_ids.update(lineup_data["home"]["batter_ids"])

    if "away" in lineup_data and "batter_ids" in lineup_data["away"]:
        all_player_ids.update(lineup_data["away"]["batter_ids"])

    # Add pitchers
    if "home" in lineup_data and "pitcher_id" in lineup_data["home"]:
        pitcher_ids = lineup_data["home"]["pitcher_id"]
        if isinstance(pitcher_ids, list):
            all_player_ids.update(pitcher_ids)
        else:
            all_player_ids.add(pitcher_ids)

    if "away" in lineup_data and "pitcher_id" in lineup_data["away"]:
        pitcher_ids = lineup_data["away"]["pitcher_id"]
        if isinstance(pitcher_ids, list):
            all_player_ids.update(pitcher_ids)
        else:
            all_player_ids.add(pitcher_ids)

    # Convert to list and fetch handedness data
    player_ids_list = list(all_player_ids)

    logging.info(f"Fetching handedness for {len(player_ids_list)} players in game")

    return fetch_player_handedness(player_ids_list)


def add_handedness_to_projections(
    projections_df: pl.DataFrame,
    handedness_df: pl.DataFrame,
    player_id_col: str = "xMLBAMID",
) -> pl.DataFrame:
    """
    Add handedness data to player projections DataFrame.

    Args:
        projections_df: DataFrame with player projections
        handedness_df: DataFrame with handedness data from fetch_player_handedness()
        player_id_col: Column name in projections_df that contains player IDs

    Returns:
        Projections DataFrame with added 'stand' and 'p_throws' columns

    """
    # Join handedness data to projections
    enhanced_df = projections_df.join(
        handedness_df.select(
            [
                pl.col("player_id"),
                pl.col("bat_side").alias("stand"),
                pl.col("pitch_hand").alias("p_throws"),
            ],
        ),
        left_on=player_id_col,
        right_on="player_id",
        how="left",
    )

    # Fill missing handedness with defaults
    enhanced_df = enhanced_df.with_columns(
        [
            pl.col("stand").fill_null("R"),
            pl.col("p_throws").fill_null("R"),
        ],
    )

    return enhanced_df


def enhance_projections_with_handedness(
    bat_projections_df: pl.DataFrame,
    pit_projections_df: pl.DataFrame,
    lineup_data: dict,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Complete pipeline to add handedness data to both batter and pitcher projections.

    Args:
        bat_projections_df: Batter projections DataFrame
        pit_projections_df: Pitcher projections DataFrame
        lineup_data: Lineup data from get_batting_orders()

    Returns:
        Tuple of (enhanced_bat_projections, enhanced_pit_projections)

    """
    # Get handedness for all players in the game
    handedness_df = get_game_handedness_data(lineup_data)

    # Add handedness to both projection DataFrames
    enhanced_bat_df = add_handedness_to_projections(bat_projections_df, handedness_df)
    enhanced_pit_df = add_handedness_to_projections(pit_projections_df, handedness_df)

    return enhanced_bat_df, enhanced_pit_df
