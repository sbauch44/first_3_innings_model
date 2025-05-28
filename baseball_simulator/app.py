import datetime

import polars as pl
import storage
import streamlit as st

# Set page config
st.set_page_config(
    page_title="Baseball Simulation Results", page_icon="⚾", layout="wide"
)

st.title("⚾ Baseball Inning Simulator Results")

# Sidebar for main controls
st.sidebar.header("📅 Date Selection")
selected_date = st.sidebar.date_input(
    "Select Date", datetime.datetime.now(tz=datetime.timezone.utc).date()
)
date_str = selected_date.strftime("%Y-%m-%d")

# Load data button
if st.sidebar.button("🔄 Load Results", type="primary"):
    with st.spinner(f"Loading results for {date_str}..."):
        results_df = storage.load_simulation_results_for_date(date_str)

        if results_df is not None and not results_df.is_empty():
            st.session_state.results_df = results_df
            st.session_state.date_loaded = date_str
            st.sidebar.success(f"✅ Loaded {len(results_df)} records")
        else:
            st.sidebar.error("❌ No simulation results found for this date.")
            if "results_df" in st.session_state:
                del st.session_state.results_df

# Check if data is loaded
if "results_df" not in st.session_state:
    st.info("👆 Please select a date and click 'Load Results' to view simulation data.")
    st.stop()

results_df = st.session_state.results_df

# Main content area
col1, col2 = st.columns([1, 3])

with col1:
    st.header("🔍 Filters")

    # Game filter
    st.subheader("🏟️ Game Selection")

    # Get unique games (assuming we can derive game info from team names or add game_pk to data)
    unique_games = (
        results_df.filter(pl.col("team") != "Total")
        .select(["team_name"])
        .unique()
        .sort("team_name")
    )

    # Create game options - pair up home/away teams
    game_options = ["All Games"]
    team_names = unique_games["team_name"].to_list()

    # If we have team pairs, create game descriptions
    if len(team_names) >= 2:
        # This is a simplified approach - you might want to enhance this based on your data structure
        games_found = []
        for i in range(0, len(team_names), 2):
            if i + 1 < len(team_names):
                game_desc = f"{team_names[i]} vs {team_names[i + 1]}"
                games_found.append(game_desc)
                game_options.append(game_desc)

    selected_game = st.selectbox("Select Game:", game_options)

    # Team filter
    st.subheader("👥 Team Selection")
    team_options = ["All Teams"] + results_df["team"].unique().sort().to_list()
    selected_team = st.selectbox("Select Team:", team_options)

    # If specific team selected, show team name
    if selected_team != "All Teams":
        team_name_filter = results_df.filter(pl.col("team") == selected_team)[
            "team_name"
        ].unique()
        if not team_name_filter.is_empty():
            st.info(f"**Team:** {team_name_filter[0]}")

    # Inning filter
    st.subheader("🔢 Inning Selection")
    inning_options = ["All Innings"] + sorted(
        [str(x) for x in results_df["inning"].unique().to_list()]
    )
    selected_inning = st.selectbox("Select Inning:", inning_options)

    # Stat filter
    st.subheader("📊 Statistic Selection")
    stat_options = ["All Stats"] + sorted(results_df["stat"].unique().to_list())
    selected_stat = st.selectbox("Select Statistic:", stat_options)

    # Probability threshold filter
    st.subheader("🎯 Probability Filter")
    min_probability = st.slider(
        "Minimum Probability:",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
        format="%.2f",
    )

with col2:
    st.header("📊 Results")

    # Apply filters
    filtered_df = results_df
    filter_description = []

    # Apply team filter
    if selected_team != "All Teams":
        filtered_df = filtered_df.filter(pl.col("team") == selected_team)
        filter_description.append(f"Team: {selected_team}")

    # Apply inning filter
    if selected_inning != "All Innings":
        filtered_df = filtered_df.filter(pl.col("inning") == int(selected_inning))
        filter_description.append(f"Inning: {selected_inning}")

    # Apply stat filter
    if selected_stat != "All Stats":
        filtered_df = filtered_df.filter(pl.col("stat") == selected_stat)
        filter_description.append(f"Stat: {selected_stat}")

    # Apply game filter (this is more complex and might need adjustment based on your data structure)
    if selected_game != "All Games" and " vs " in selected_game:
        teams_in_game = selected_game.split(" vs ")
        filtered_df = filtered_df.filter(
            pl.col("team_name").is_in(teams_in_game) | (pl.col("team") == "Total")
        )
        filter_description.append(f"Game: {selected_game}")

    # Apply probability filter
    if min_probability > 0:
        filtered_df = filtered_df.filter(pl.col("probability") >= min_probability)
        filter_description.append(f"Min Probability: {min_probability:.2%}")

    # Display filter summary
    if filter_description:
        st.info("🔍 **Active Filters:** " + " | ".join(filter_description))

    # Display results count
    st.metric("📋 Filtered Results", len(filtered_df))

    if not filtered_df.is_empty():
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📋 Data Table",
                "📊 Summary Stats",
                "🎲 Probability Heatmap",
                "💰 Odds View",
            ]
        )

        with tab1:
            st.subheader("Detailed Results")

            # Format the dataframe for better display
            display_df = filtered_df.with_columns(
                [
                    (pl.col("probability") * 100).round(2).alias("probability_pct"),
                    pl.col("decimal_odds").round(2),
                ]
            ).select(
                [
                    "inning",
                    "team",
                    "team_name",
                    "stat",
                    "number_bin",
                    "probability_pct",
                    "decimal_odds",
                    "american_odds",
                ]
            )

            # Rename columns for display
            display_df = display_df.rename(
                {
                    "probability_pct": "Probability (%)",
                    "decimal_odds": "Decimal Odds",
                    "american_odds": "American Odds",
                    "number_bin": "Count",
                    "team_name": "Team Name",
                }
            )

            st.dataframe(display_df, use_container_width=True, hide_index=True)

        with tab2:
            st.subheader("Summary Statistics")

            # Group by key dimensions
            summary_stats = (
                filtered_df.group_by(["team", "stat"])
                .agg(
                    [
                        pl.col("probability").mean().alias("avg_probability"),
                        pl.col("probability").max().alias("max_probability"),
                        pl.col("decimal_odds").mean().alias("avg_decimal_odds"),
                        pl.len().alias("count"),
                    ]
                )
                .sort(["team", "stat"])
            )

            if not summary_stats.is_empty():
                st.dataframe(summary_stats, use_container_width=True)
            else:
                st.info("No summary statistics available for current filters.")

        with tab3:
            st.subheader("Probability Heatmap")

            try:
                # Create pivot table for heatmap
                pivot_df = (
                    filtered_df.select(
                        ["inning", "team", "stat", "number_bin", "probability"]
                    )
                    .pivot(
                        index=["inning", "team", "stat"],
                        on="number_bin",
                        values="probability",
                    )
                    .fill_null(0)
                )

                if not pivot_df.is_empty():
                    st.dataframe(pivot_df, use_container_width=True)

                    # If matplotlib/seaborn available, could add actual heatmap visualization here
                    st.info(
                        "💡 **Tip:** Higher probabilities indicate more likely outcomes."
                    )
                else:
                    st.info("No data available for heatmap with current filters.")

            except Exception as e:
                st.error(f"Error creating heatmap: {str(e)}")

        with tab4:
            st.subheader("Betting Odds View")

            # Filter for reasonable odds (exclude extreme values)
            odds_df = (
                filtered_df.filter(
                    (pl.col("decimal_odds").is_not_null())
                    & (pl.col("decimal_odds") > 1)
                    & (pl.col("decimal_odds") < 1000)
                )
                .select(
                    [
                        "inning",
                        "team",
                        "team_name",
                        "stat",
                        "number_bin",
                        "probability",
                        "decimal_odds",
                        "american_odds",
                    ]
                )
                .with_columns(
                    [(pl.col("probability") * 100).round(2).alias("probability_pct")]
                )
            )

            if not odds_df.is_empty():
                # Sort by most favorable odds
                odds_df = odds_df.sort("decimal_odds")

                st.dataframe(
                    odds_df.select(
                        [
                            "team_name",
                            "stat",
                            "number_bin",
                            "inning",
                            "probability_pct",
                            "decimal_odds",
                            "american_odds",
                        ]
                    ).rename(
                        {
                            "team_name": "Team",
                            "stat": "Stat",
                            "number_bin": "Count",
                            "inning": "Inning",
                            "probability_pct": "Probability (%)",
                            "decimal_odds": "Decimal Odds",
                            "american_odds": "American Odds",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                st.info("💡 **Lower decimal odds indicate higher probability events.**")
            else:
                st.info("No valid odds data available with current filters.")

    else:
        st.warning(
            "🔍 No results match your current filter criteria. Try adjusting the filters."
        )

# Footer with additional info
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown("""
This app displays baseball simulation results with the following statistics:
- **H**: Hits
- **R**: Runs  
- **BB**: Walks
- **HR**: Home Runs

**Teams:**
- Individual team results
- Combined totals for both teams
""")

if "date_loaded" in st.session_state:
    st.sidebar.success(f"📅 Data loaded for: {st.session_state.date_loaded}")
