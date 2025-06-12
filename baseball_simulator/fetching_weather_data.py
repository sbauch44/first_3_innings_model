import logging
import time
from typing import Dict, List, Optional, Tuple

import polars as pl
import statsapi

# Set up logger
logger = logging.getLogger("mlb_data_extractor")
logger.setLevel(logging.INFO)

# Create console handler with formatting
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Known domed/retractable roof stadiums
DOMED_STADIUMS = {
    "Rogers Centre",  # Toronto Blue Jays - retractable roof
    "Tropicana Field",  # Tampa Bay Rays - fixed dome
    "Minute Maid Park",  # Houston Astros - retractable roof
    "Chase Field",  # Arizona Diamondbacks - retractable roof
    "T-Mobile Park",  # Seattle Mariners - retractable roof
    "Marlins Park",  # Miami Marlins - retractable roof
    "Globe Life Field",  # Texas Rangers - retractable roof
}


def extract_game_data(game_id: int) -> Optional[Dict]:
    """
    Extract relevant game data from MLB statsapi for a single game.

    Args:
        game_id: MLB game ID

    Returns:
        Dictionary containing extracted data or None if extraction fails
    """
    try:
        # Get boxscore data
        boxscore = statsapi.boxscore_data(game_id)

        # Extract runs for both teams
        away_runs = (
            boxscore.get("away", {})
            .get("teamStats", {})
            .get("batting", {})
            .get("runs", 0)
        )
        home_runs = (
            boxscore.get("home", {})
            .get("teamStats", {})
            .get("batting", {})
            .get("runs", 0)
        )
        total_runs = away_runs + home_runs

        # Extract team names
        away_team = boxscore.get("away", {}).get("team", {}).get("name", "Unknown")
        home_team = boxscore.get("home", {}).get("team", {}).get("name", "Unknown")

        # Extract team_ids
        away_team_id = boxscore.get("away", {}).get("team", {}).get("id", None)
        home_team_id = boxscore.get("home", {}).get("team", {}).get("id", None)

        # Extract game box info
        game_box_info = boxscore.get("gameBoxInfo", [])

        # Initialize extracted values
        weather = None
        wind = None
        venue = None

        # Parse gameBoxInfo to extract weather, wind, and venue
        for info in game_box_info:
            label = info.get("label", "").lower()
            value = info.get("value", "")

            if label == "weather":
                weather = value
            elif label == "wind":
                wind = value
            elif label == "venue":
                venue = value

        # Create data record
        game_data = {
            "game_id": game_id,
            "away_team": away_team,
            "away_team_id": away_team_id,
            "home_team": home_team,
            "home_team_id": home_team_id,
            "away_runs": away_runs,
            "home_runs": home_runs,
            "total_runs": total_runs,
            "weather": weather,
            "wind": wind,
            "venue": venue,
        }

        return game_data

    except Exception as e:
        logger.error(f"Error extracting data for game {game_id}: {str(e)}")
        return None


def parse_weather_data(
    weather_str: str,
) -> Tuple[Optional[float], Optional[str], Optional[bool], Optional[bool]]:
    """
    Parse weather string to extract temperature, conditions, and stadium info.

    Args:
        weather_str: Weather string like "68 degrees, Roof Closed." or "82 degrees, Partly Cloudy."

    Returns:
        Tuple of (temperature, conditions, is_domed_stadium, roof_closed)
    """
    if not weather_str:
        return None, None, None, None

    try:
        # Extract temperature
        temp = None
        if "degrees" in weather_str.lower():
            temp_part = weather_str.split("degrees")[0].strip()
            # Extract numeric part
            temp_numeric = "".join(filter(str.isdigit, temp_part.split()[-1]))
            if temp_numeric:
                temp = float(temp_numeric)

        # Extract conditions (everything after the comma)
        conditions = None
        if "," in weather_str:
            conditions = weather_str.split(",", 1)[1].strip().rstrip(".")

        # Determine if it's a domed stadium and roof status
        is_domed = None
        roof_closed = None

        conditions_lower = conditions.lower() if conditions else ""

        if "roof closed" in conditions_lower:
            is_domed = True
            roof_closed = True
        elif "roof open" in conditions_lower:
            is_domed = True
            roof_closed = False
        else:
            # For open stadiums, conditions will be weather descriptions
            is_domed = False
            roof_closed = None

        return temp, conditions, is_domed, roof_closed

    except Exception:
        return None, None, None, None


def parse_wind_data(
    wind_str: str,
) -> Tuple[Optional[float], Optional[str], Optional[bool]]:
    """
    Parse wind string to extract speed, direction, and if it affects play.

    Args:
        wind_str: Wind string like "0 mph, None." or "7 mph, In From CF."

    Returns:
        Tuple of (wind_speed, wind_direction, affects_play)
    """
    if not wind_str:
        return None, None, None

    try:
        # Extract wind speed
        wind_speed = None
        if "mph" in wind_str.lower():
            speed_part = wind_str.split("mph")[0].strip()
            # Extract numeric part
            speed_numeric = "".join(
                filter(lambda x: x.isdigit() or x == ".", speed_part.split()[-1])
            )
            if speed_numeric:
                wind_speed = float(speed_numeric)

        # Extract direction (everything after the comma)
        direction = None
        if "," in wind_str:
            direction = wind_str.split(",", 1)[1].strip().rstrip(".")

        # Determine if wind affects play (non-zero speed and not "None")
        affects_play = False
        if wind_speed and wind_speed > 0 and direction and direction.lower() != "none":
            affects_play = True

        # Standardize common wind directions
        if direction:
            direction_lower = direction.lower()
            if "in from" in direction_lower:
                # Wind blowing in (reduces home runs)
                if "cf" in direction_lower or "center" in direction_lower:
                    direction = "In From CF"
                elif "lf" in direction_lower or "left" in direction_lower:
                    direction = "In From LF"
                elif "rf" in direction_lower or "right" in direction_lower:
                    direction = "In From RF"
            elif "out to" in direction_lower:
                # Wind blowing out (increases home runs)
                if "cf" in direction_lower or "center" in direction_lower:
                    direction = "Out To CF"
                elif "lf" in direction_lower or "left" in direction_lower:
                    direction = "Out To LF"
                elif "rf" in direction_lower or "right" in direction_lower:
                    direction = "Out To RF"
            elif "l to r" in direction_lower or "left to right" in direction_lower:
                direction = "L To R"
            elif "r to l" in direction_lower or "right to left" in direction_lower:
                direction = "R To L"

        return wind_speed, direction, affects_play

    except Exception:
        return None, None, None


def process_games(game_ids: List[int], output_file: str = "mlb_game_data.parquet"):
    """
    Process multiple games and save to parquet file.

    Args:
        game_ids: List of MLB game IDs to process
        output_file: Output parquet file name
    """
    all_data = []

    logger.info(f"Starting to process {len(game_ids)} games")

    for i, game_id in enumerate(game_ids):
        logger.info(f"Processing game {i + 1}/{len(game_ids)}: {game_id}")

        # Extract basic game data
        game_data = extract_game_data(game_id)

        if game_data:
            # Parse weather data
            temp, weather_conditions, is_domed, roof_closed = parse_weather_data(
                game_data.get("weather")
            )

            # Use venue knowledge to determine if stadium is domed (if not clear from weather)
            if is_domed is None and game_data.get("venue") in DOMED_STADIUMS:
                is_domed = True
            elif is_domed is None:
                is_domed = False

            # Parse wind data
            wind_speed, wind_direction, wind_affects_play = parse_wind_data(
                game_data.get("wind")
            )

            # Create enhanced record
            enhanced_data = {
                "game_id": game_data["game_id"],
                "away_team": game_data["away_team"],
                "away_team_id": game_data["away_team_id"],
                "home_team": game_data["home_team"],
                "home_team_id": game_data["home_team_id"],
                "away_runs": game_data["away_runs"],
                "home_runs": game_data["home_runs"],
                "total_runs": game_data["total_runs"],
                "venue": game_data["venue"],
                "weather_raw": game_data["weather"],
                "wind_raw": game_data["wind"],
                "temperature": temp,
                "weather_conditions": weather_conditions,
                "is_domed_stadium": is_domed,
                "roof_closed": roof_closed,
                "wind_speed_mph": wind_speed,
                "wind_direction": wind_direction,
                "wind_affects_play": wind_affects_play,
            }

            all_data.append(enhanced_data)
        else:
            logger.warning(f"Failed to extract data for game {game_id}")

        # Add a small delay to be respectful to the API
        time.sleep(0.1)

    # Create Polars DataFrame
    df = pl.DataFrame(
        all_data,
        schema={
            "game_id": pl.Int64,
            "away_team": pl.Utf8,
            "away_team_id": pl.Int64,
            "home_team": pl.Utf8,
            "home_team_id": pl.Int64,
            "away_runs": pl.Int64,
            "home_runs": pl.Int64,
            "total_runs": pl.Int64,
            "venue": pl.Utf8,
            "weather_raw": pl.Utf8,
            "wind_raw": pl.Utf8,
            "temperature": pl.Utf8,
            "weather_conditions": pl.Utf8,
            "is_domed_stadium": pl.Utf8,
            "roof_closed": pl.Utf8,
            "wind_speed_mph": pl.Utf8,
            "wind_direction": pl.Utf8,
            "wind_affects_play": pl.Utf8,
        },
    )

    # Save to parquet
    df.write_parquet(output_file)
    logger.info(f"Saved {len(df)} games to {output_file}")

    # Display summary
    logger.info("Data Summary:")
    logger.info(f"Total games: {len(df)}")
    logger.info(f"Average total runs: {df['total_runs'].mean():.2f}")
    logger.info(f"Unique venues: {df['venue'].n_unique()}")

    # Check if temperature data exists
    temp_data = df.filter(pl.col("temperature").is_not_null())
    if len(temp_data) > 0:
        temp_min = temp_data["temperature"].min()
        temp_max = temp_data["temperature"].max()
        logger.info(f"Temperature range: {temp_min:.0f}°F - {temp_max:.0f}°F")

    # Stadium type breakdown
    domed_games = df.filter(pl.col("is_domed_stadium")).height
    logger.info(
        f"Games in domed stadiums: {domed_games} ({domed_games / len(df) * 100:.1f}%)"
    )

    # Wind analysis
    windy_games = df.filter(pl.col("wind_affects_play")).height
    logger.info(
        f"Games with significant wind: {windy_games} ({windy_games / len(df) * 100:.1f}%)"
    )

    # Wind direction breakdown
    wind_data = df.filter(pl.col("wind_direction").is_not_null())
    if len(wind_data) > 0:
        logger.info("Most common wind directions:")
        wind_counts = (
            wind_data.group_by("wind_direction")
            .count()
            .sort("count", descending=True)
            .head(5)
        )

        for row in wind_counts.iter_rows(named=True):
            logger.info(f"  {row['wind_direction']}: {row['count']} games")

    # Weather conditions breakdown
    weather_data = df.filter(pl.col("weather_conditions").is_not_null())
    if len(weather_data) > 0:
        logger.info("Weather conditions:")
        weather_counts = (
            weather_data.group_by("weather_conditions")
            .count()
            .sort("count", descending=True)
            .head(5)
        )

        for row in weather_counts.iter_rows(named=True):
            logger.info(f"  {row['weather_conditions']}: {row['count']} games")

    return df


# Helper function to get game IDs for a date range
def get_game_ids_for_date_range(start_date: str, end_date: str) -> List[int]:
    """
    Get game IDs for a date range.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format

    Returns:
        List of game IDs
    """
    try:
        schedule = statsapi.schedule(start_date=start_date, end_date=end_date)
        game_ids = [game["game_id"] for game in schedule]
        logger.info(f"Found {len(game_ids)} games between {start_date} and {end_date}")
        return game_ids
    except Exception as e:
        logger.error(f"Error getting schedule: {str(e)}")
        return []


# Exampl
# Example of how to get game IDs for a date range
def main():
    game_ids = get_game_ids_for_date_range("2023-03-01", "2024-12-07")
    logger.info(f"Processing {len(game_ids)} games...")

    df = process_games(game_ids, "../raw_data/weather_data.parquet")

    logger.info("First few rows of the dataset:")
    logger.info(f"\n{df.head()}")


if __name__ == "__main__":
    main()
