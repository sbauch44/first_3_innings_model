import pandas as pd
import polars as pl
import requests
import re
import json
from pybaseball import (
    statcast,
    statcast_fielding,
)


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
    else: 
        ValueError("No data found in the response.")
        return None


def fetch_defensive_stats(YEAR):
    df_tmp = statcast_fielding.statcast_outs_above_average(year=YEAR, pos="ALL")
    pdf = pl.from_pandas(df_tmp)
    if pdf.is_empty():
        raise ValueError("No data found for the specified year.")
    return pdf


def fetch_fangraphs_projections(position=None):
    if position not in ['pit', 'bat']:
        raise ValueError("Position must be 'pit' or 'bat'")
    
    url = 'https://www.fangraphs.com/api/projections?type=steamerr&stats={position}&pos=all&team=0&players=0&lg=all&z=1745190960732&download=1'
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code}")
    df = pl.DataFrame(response.json())
    if df.is_empty():
        raise ValueError("No data found in the response.")
    return df


def get_batting_orders(game_json):
    batting_orders = {}
    boxscores = game_json["liveData"]["boxscore"]["teams"]
    home = boxscores["home"]["battingOrder"]
    away = boxscores["away"]["battingOrder"]
    batting_orders["home"] = home
    batting_orders["away"] = away

    return batting_orders