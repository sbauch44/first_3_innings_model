import pandas as pd
import polars as pl
import requests
import re
import json
import statsapi
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
    cookies = {
        'fg_uuid': '178ccbc1-6901-4f69-8300-8bf3c1248d98',
        'usprivacy': '1N--',
        'wordpress_logged_in_0cae6f5cb929d209043cb97f8c2eee44': 'sb4422%7C1776720185%7CeRCHOYBE0Q23GmAO3ptRfFSgAT9cstCs9FQsaJdgIdY%7C970d7f02f4b7640c9057580d54edfbcdc20e73bcc2a2be00fc3c429cf9c0b448',
        'wp_automatewoo_visitor_0cae6f5cb929d209043cb97f8c2eee44': 'gq3g22frpm7sstlukqjb',
        'fg_is_member': 'true',
    }
    headers = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'accept-language': 'en-US,en;q=0.9',
        'cache-control': 'max-age=0',
        'priority': 'u=0, i',
        'sec-ch-ua': '"Chromium";v="136", "Google Chrome";v="136", "Not.A/Brand";v="99"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36',
    }
    params = {
        'type': 'steamerr',
        'stats': position,
        'pos': 'all',
        'team': '0',
        'players': '0',
        'lg': 'all',
        'z': '1747738554',
        'download': '1',
    }
    response = requests.get('https://www.fangraphs.com/api/projections', params=params, cookies=cookies, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code}")
    df = pl.DataFrame(response.json())
    if df.is_empty():
        raise ValueError("No data found in the response.")
    return df


def get_batting_orders(game_pk):
    game_json = statsapi.get('game',{'gamePk': game_pk})
    batting_orders = {}
    boxscores = game_json["liveData"]["boxscore"]["teams"]
    home = boxscores["home"]["battingOrder"]
    away = boxscores["away"]["battingOrder"]
    batting_orders["home"] = home
    batting_orders["away"] = away
    return batting_orders


def get_game_info(game_pk):
    sched = statsapi.schedule()
    game_dict = next((game for game in sched if game['game_id'] == game_pk), None)
    return game_dict