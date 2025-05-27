import json
import re

import polars as pl
import requests
import statsapi
from pybaseball import (
    statcast,
    statcast_fielding,
)

import config


def fetch_statcast_data():
    data = statcast(verbose=True)
    pl_data = pl.from_pandas(data)
    if pl_data.is_empty():
        raise ValueError("No data found in the response.")
    return pl_data


def fetch_park_factors():
    url = "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors?type=venue&year=2025&batSide=&stat=index_wOBA&condition=All&rolling=3&parks=mlb"

    response = requests.get(url)
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


def fetch_defensive_stats(YEAR):
    df_tmp = statcast_fielding.statcast_outs_above_average(year=YEAR, pos="ALL")
    pdf = pl.from_pandas(df_tmp)
    if pdf.is_empty():
        raise ValueError("No data found for the specified year.")
    return pdf


def fetch_fangraphs_projections(position=None, cookies=config.FANGRAPHS_COOKIES, headers=config.FANGRAPHS_HEADERS):
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
    response = requests.get("https://www.fangraphs.com/api/projections", params=params, cookies=cookies, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code}")
    df = pl.DataFrame(response.json())
    if df.is_empty():
        raise ValueError("No data found in the response.")
    return df


def get_batting_orders(game_pk):
    game_json = statsapi.get("game",{"gamePk": game_pk})
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
