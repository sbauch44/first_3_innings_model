import logging
import random
from typing import Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BaseballSimulator:
    def __init__(
        self,
        idata,
        scaler,
        outcome_labels,
        predictor_cols,
        continuous_cols,
        categorical_cols,
        league_avg_rates,
    ):
        """Initialize with performance optimizations."""
        self.idata = idata
        self.scaler = scaler
        self.outcome_labels = outcome_labels
        self.n_categories = len(self.outcome_labels)
        self.predictor_cols = predictor_cols
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        self.league_avg_rates = league_avg_rates

        # PRE-COMPUTE INDICES FOR FAST ARRAY ACCESS
        self._setup_column_indices()

        # Pre-calculate mean posterior parameters
        logger.info("Pre-calculating mean posterior parameters")
        try:
            self.mean_intercepts_val = (
                self.idata.posterior["intercepts"]
                .mean(dim=("chain", "draw"))
                .to_numpy()
            )
            self.mean_betas_val = (
                self.idata.posterior["betas"].mean(dim=("chain", "draw")).to_numpy()
            )
            self.n_model_predictors = self.mean_betas_val.shape[0]

            # Handle shape mismatches once at initialization
            self._fix_parameter_shapes()

            logger.info("Mean posterior parameters pre-calculated successfully.")
        except Exception as e:
            logger.error(f"Could not pre-calculate parameters: {e}")
            raise

    def _setup_column_indices(self):
        """Pre-compute column indices for fast array access."""
        # Create mapping from column name to index in predictor_cols
        self.col_to_idx = {col: idx for idx, col in enumerate(self.predictor_cols)}

        # Pre-compute indices for continuous and categorical columns
        self.continuous_indices = [self.col_to_idx[col] for col in self.continuous_cols]
        self.categorical_indices = [
            self.col_to_idx[col] for col in self.categorical_cols
        ]

        # Pre-allocate arrays for reuse
        self.input_array = np.zeros(len(self.predictor_cols))
        self.continuous_temp = np.zeros(len(self.continuous_cols))
        self.categorical_temp = np.zeros(len(self.categorical_cols))

        logger.info(
            f"Column indices pre-computed: {len(self.continuous_indices)} continuous, {len(self.categorical_indices)} categorical"
        )

    def _fix_parameter_shapes(self):
        """Fix parameter shapes once at initialization."""
        expected_intercept_shape = (self.n_categories,)
        expected_beta_shape = (self.n_model_predictors, self.n_categories)

        if self.mean_intercepts_val.shape != expected_intercept_shape:
            if self.mean_intercepts_val.shape == (self.n_categories - 1,):
                self.mean_intercepts_val = np.concatenate(
                    [self.mean_intercepts_val, [0.0]]
                )
            else:
                raise ValueError("Cannot resolve intercept shape mismatch")

        if self.mean_betas_val.shape != expected_beta_shape:
            if self.mean_betas_val.shape == (
                self.n_model_predictors,
                self.n_categories - 1,
            ):
                ref_betas = np.zeros((self.n_model_predictors, 1))
                self.mean_betas_val = np.concatenate(
                    [self.mean_betas_val, ref_betas], axis=1
                )
            else:
                raise ValueError("Cannot resolve beta shape mismatch")

    def predict_pa_outcome_probs_fast(
        self, pa_inputs_dict: dict[str, float]
    ) -> np.ndarray:
        """Optimized prediction function avoiding DataFrame creation."""
        # Fill input array directly using pre-computed indices
        for col, value in pa_inputs_dict.items():
            if col in self.col_to_idx:
                self.input_array[self.col_to_idx[col]] = value

        # Extract continuous and categorical data using pre-computed indices
        for i, idx in enumerate(self.continuous_indices):
            self.continuous_temp[i] = self.input_array[idx]

        for i, idx in enumerate(self.categorical_indices):
            self.categorical_temp[i] = self.input_array[idx]

        # Scale continuous features (reshape for scaler)
        continuous_scaled = self.scaler.transform(self.continuous_temp.reshape(1, -1))

        # Combine features
        X_new = np.concatenate([continuous_scaled.flatten(), self.categorical_temp])

        # Calculate linear predictor
        mu_mean = self.mean_intercepts_val + X_new @ self.mean_betas_val

        # Apply softmax with numerical stability
        exp_mu = np.exp(mu_mean - np.max(mu_mean))
        probabilities = exp_mu / np.sum(exp_mu)

        return probabilities

    def simulate_single_inning_fast(
        self,
        inning_num: int,
        is_top_inning: bool,
        lineup: list[dict[str, Any]],
        start_batter_idx: int,
        pitcher_inputs: dict[str, Any],
        game_context: dict[str, Any],
    ) -> tuple[int, int, int, int, int]:
        """Optimized inning simulation with reduced allocations."""
        outs = 0
        hits = runs = walks = home_runs = 0
        bases = [0, 0, 0]
        current_batter_idx = start_batter_idx
        lineup_len = len(lineup)

        # Pre-compute context that doesn't change during inning
        if is_top_inning:
            fielding_team_defense_rating = game_context["home_team_defense_rating"]
            is_batter_home = 0
        else:
            fielding_team_defense_rating = game_context["away_team_defense_rating"]
            is_batter_home = 1

        # Pre-build base context dict (reused for each PA)
        base_pa_inputs = {
            **pitcher_inputs,
            "park_factor_input": game_context["park_factor_input"],
            "team_defense_oaa_input": fielding_team_defense_rating,
            "is_batter_home": is_batter_home,
        }

        # Pre-compute pitcher handedness for platoon advantage
        pitcher_throws = pitcher_inputs.get("p_throws", "R")

        # Cache random values for common probabilities
        rate_1st_to_3rd = self.league_avg_rates["rate_1st_to_3rd_on_single"]
        rate_score_from_1st_double = self.league_avg_rates[
            "rate_score_from_1st_on_double"
        ]
        gidp_rate = self.league_avg_rates.get("gidp_rate_if_gb_opportunity", 0.13)

        # Pre-allocate outcome array for random choice
        possible_outcomes = np.array(list(self.outcome_labels.keys()))

        while outs < 3:
            batter_spot = current_batter_idx % lineup_len
            current_batter = lineup[batter_spot]

            # Build PA inputs efficiently
            pa_inputs = {**base_pa_inputs, **current_batter}

            # Calculate platoon advantage inline
            batter_stand = current_batter.get("stand", "R")
            pa_inputs["is_platoon_adv"] = (
                1
                if (
                    (batter_stand == "L" and pitcher_throws == "R")
                    or (batter_stand == "R" and pitcher_throws == "L")
                )
                else 0
            )

            # Get outcome probabilities
            outcome_probs = self.predict_pa_outcome_probs_fast(pa_inputs)

            # Sample outcome
            simulated_outcome_code = np.random.choice(
                possible_outcomes, p=outcome_probs
            )
            outcome_label = self.outcome_labels[simulated_outcome_code]

            # Process outcome with minimal allocations
            new_bases = bases.copy()  # Only copy when needed
            runs_this_pa = pa_hit = pa_walk = pa_hr = 0
            outs_before_pa = outs

            # Use elif chain for mutually exclusive outcomes
            if outcome_label == "Strikeout":
                outs += 1
            elif outcome_label == "Walk":
                pa_walk = 1
                # Inline force advancement logic
                if bases[0]:
                    if bases[1]:
                        if bases[2]:
                            runs_this_pa = 1
                        new_bases[2] = 1
                    new_bases[1] = 1
                new_bases[0] = 1
            elif outcome_label == "HBP":
                # Same as walk
                if bases[0]:
                    if bases[1]:
                        if bases[2]:
                            runs_this_pa = 1
                        new_bases[2] = 1
                    new_bases[1] = 1
                new_bases[0] = 1
            elif outcome_label == "Single":
                pa_hit = 1
                runs_this_pa = bases[2] + bases[1]  # Runners from 2nd and 3rd score

                # Handle runner from 1st
                if bases[0]:
                    if random.random() < rate_1st_to_3rd:
                        new_bases = [1, 0, 1]  # Batter to 1st, runner to 3rd
                    else:
                        new_bases = [1, 1, 0]  # Batter to 1st, runner to 2nd
                else:
                    new_bases = [1, 0, 0]  # Just batter to 1st

            elif outcome_label == "Double":
                pa_hit = 1
                runs_this_pa = bases[2] + bases[1]  # Runners from 2nd and 3rd score

                # Handle runner from 1st
                if bases[0]:
                    if random.random() < rate_score_from_1st_double:
                        runs_this_pa += 1
                        new_bases = [0, 1, 0]  # Batter to 2nd
                    else:
                        new_bases = [0, 1, 1]  # Batter to 2nd, runner to 3rd
                else:
                    new_bases = [0, 1, 0]  # Batter to 2nd

            elif outcome_label == "Triple":
                pa_hit = 1
                runs_this_pa = sum(bases)
                new_bases = [0, 0, 1]

            elif outcome_label == "HomeRun":
                pa_hit = pa_hr = 1
                runs_this_pa = 1 + sum(bases)
                new_bases = [0, 0, 0]

            else:  # Out_In_Play
                outs += 1
                # GIDP check
                if bases[0] and outs_before_pa < 2 and random.random() < gidp_rate:
                    outs = min(outs + 1, 3)  # Double play, but don't exceed 3 outs
                    new_bases = [0, 0, bases[2]]  # Simple GIDP logic
                else:
                    # Regular out with advancement
                    new_bases = [0, bases[0], bases[1] or bases[2]]

            # Update state
            runs += runs_this_pa
            hits += pa_hit
            walks += pa_walk
            home_runs += pa_hr
            bases = new_bases
            current_batter_idx += 1

        return hits, runs, walks, home_runs, (current_batter_idx % lineup_len)

    def simulate_first_three_innings(
        self,
        home_lineup: list[dict[str, Any]],
        away_lineup: list[dict[str, Any]],
        home_pitcher_inputs: dict[str, Any],
        away_pitcher_inputs: dict[str, Any],
        game_context: dict[str, Any],
    ) -> dict[str, dict[str, dict[str, int]]]:
        """Optimized three-inning simulation."""
        results = {}
        away_batter_idx = home_batter_idx = 0

        for inning in range(1, 4):
            # Top of inning (away team batting)
            away_stats = self.simulate_single_inning_fast(
                inning,
                True,
                away_lineup,
                away_batter_idx,
                home_pitcher_inputs,
                game_context,
            )
            away_batter_idx = away_stats[4]  # Update batter index

            # Bottom of inning (home team batting)
            home_stats = self.simulate_single_inning_fast(
                inning,
                False,
                home_lineup,
                home_batter_idx,
                away_pitcher_inputs,
                game_context,
            )
            home_batter_idx = home_stats[4]  # Update batter index

            results[f"inning_{inning}"] = {
                "away": {
                    "H": away_stats[0],
                    "R": away_stats[1],
                    "BB": away_stats[2],
                    "HR": away_stats[3],
                },
                "home": {
                    "H": home_stats[0],
                    "R": home_stats[1],
                    "BB": home_stats[2],
                    "HR": home_stats[3],
                },
            }

        return results

    def run_multiple_simulations(
        self,
        home_lineup: list[dict[str, Any]],
        away_lineup: list[dict[str, Any]],
        home_pitcher_inputs: dict[str, Any],
        away_pitcher_inputs: dict[str, Any],
        game_context: dict[str, Any],
        num_sims: int = 10000,
    ) -> list[dict[str, dict[str, dict[str, int]]]]:
        """Optimized batch simulation."""

        # Pre-allocate list for better memory performance
        all_results: list[dict[str, dict[str, dict[str, int]]]] = [
            {} for _ in range(num_sims)
        ]

        logger.info(f"Running {num_sims} game simulations...")
        for i in tqdm(range(num_sims)):
            all_results[i] = self.simulate_first_three_innings(
                home_lineup,
                away_lineup,
                home_pitcher_inputs,
                away_pitcher_inputs,
                game_context,
            )

        return all_results

    # Keep other methods unchanged for compatibility
    def predict_pa_outcome_probs(self, pa_inputs_dict: dict[str, float]) -> np.ndarray:
        """Backwards compatibility - delegates to fast version."""
        return self.predict_pa_outcome_probs_fast(pa_inputs_dict)

    def simulate_single_inning(self, *args, **kwargs):
        """Backwards compatibility - delegates to fast version."""
        return self.simulate_single_inning_fast(*args, **kwargs)

    def analyze_simulation_results(self, all_results: list[dict]) -> dict[str, dict]:
        """
        Analyze simulation results to calculate probabilities.

        Args:
            all_results: List of simulation results from run_multiple_simulations

        Returns:
            Dictionary with probability distributions for different outcomes
            Structure: {inning_str: {"away": {"H": {bin: prob}, "R": {...}}, "home": {...}}}

        """
        prob_dict = {}

        # Extract unique inning numbers
        innings = set()
        for result in all_results:
            innings.update(result.keys())

        for inning in sorted(innings):
            inning_dict = {"away": {}, "home": {}}

            for team in ["away", "home"]:
                # Count occurrences for different stats
                for stat in ["H", "R", "BB", "HR"]:
                    counts = {}
                    for sim_result in all_results:
                        if inning in sim_result and team in sim_result[inning]:
                            # Extract the value for this stat (default to 0 if not present)
                            val = sim_result[inning][team].get(stat, 0)

                            # Bin values of 5 or more into a "5+" bin
                            bin_key = str(val) if val < 5 else "5+"

                            if bin_key not in counts:
                                counts[bin_key] = 0
                            counts[bin_key] += 1

                    # Calculate probabilities
                    total = sum(counts.values())
                    prob_bins = {
                        bin_key: count / total for bin_key, count in counts.items()
                    }

                    # Make sure all bins from 0 to 4 and 5+ exist, even if probability is 0
                    for bin_key in ["0", "1", "2", "3", "4", "5+"]:
                        if bin_key not in prob_bins:
                            prob_bins[bin_key] = 0.0

                    inning_dict[team][stat] = prob_bins

            prob_dict[inning] = inning_dict

        return prob_dict

    def convert_probabilities_to_dataframe(self, prob_dict: dict) -> pd.DataFrame:
        """
        Convert the probability dictionary to a pandas DataFrame for easier analysis.

        Args:
            prob_dict: Dictionary from analyze_simulation_results

        Returns:
            pandas.DataFrame with columns: inning, team, stat, number, probability

        """
        data_for_df = []

        for inn_str, teams_data in prob_dict.items():
            # Extract inning number from the key
            try:
                inning_num = (
                    int(inn_str.split("_")[1]) if "_" in inn_str else int(inn_str)
                )
            except (IndexError, ValueError):
                logger.warning(
                    f"Warning: Could not parse inning number from key '{inn_str}'. Using as-is.",
                )
                inning_num = inn_str

            for team, stats_data in teams_data.items():
                for stat, bins_data in stats_data.items():
                    for number_bin, probability in bins_data.items():
                        data_for_df.append(
                            {
                                "inning": inning_num,
                                "team": team,
                                "stat": stat,
                                "number": number_bin,  # Keep as string ('0', '1', ..., '5+')
                                "probability": probability,
                            },
                        )

        # Create DataFrame
        if data_for_df:
            return pd.DataFrame(data_for_df)
        # Return empty DataFrame with correct columns if no data
        return pd.DataFrame(columns=["inning", "team", "stat", "number", "probability"])

    def visualize_results(
        self,
        prob_dict: dict,
        output_file: Optional[str] = None,
    ) -> None:
        """
        Create visualizations of simulation results.

        Args:
            prob_dict: Dictionary from analyze_simulation_results
            output_file: Optional path to save the visualization (if None, will display)

        """
        # This is a placeholder method - in a real implementation, you would use
        # matplotlib, seaborn, or another visualization library to create charts
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Convert to DataFrame for easier plotting
            df = self.convert_probabilities_to_dataframe(prob_dict)

            # Set up the figure
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            fig.suptitle("Baseball Simulation Results", fontsize=16)

            # Flatten for easier indexing
            axes = axes.flatten()

            stats = ["H", "R", "BB", "HR"]
            teams = ["away", "home"]

            for i, stat in enumerate(stats):
                for j, team in enumerate(teams):
                    ax_idx = i * 2 + j

                    # Filter data
                    plot_data = df[(df["stat"] == stat) & (df["team"] == team)]

                    if not plot_data.empty:
                        # Pivot for heatmap format
                        pivot_data = plot_data.pivot(
                            index="inning",
                            columns="number",
                            values="probability",
                        ).fillna(0)

                        # Plot
                        sns.heatmap(
                            pivot_data,
                            annot=True,
                            fmt=".2f",
                            cmap="YlGnBu",
                            cbar=False,
                            ax=axes[ax_idx],
                        )
                        axes[ax_idx].set_title(f"{team.capitalize()} Team - {stat}")
                        axes[ax_idx].set_ylabel("Inning")
                        axes[ax_idx].set_xlabel("Count")

            plt.tight_layout(rect=(0, 0, 1, 0.96))

            if output_file:
                plt.savefig(output_file)
                print(f"Visualization saved to {output_file}")
            else:
                plt.show()

        except ImportError:
            print("Visualization requires matplotlib and seaborn packages.")
            return
