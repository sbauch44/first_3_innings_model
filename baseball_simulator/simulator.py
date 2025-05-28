import logging
import random
from typing import Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BaseballSimulator:
    """
    A simulator for baseball games using a Bayesian model for plate appearance outcomes.

    This class loads a trained PyMC model and provides methods to simulate
    plate appearances, innings, and partial games.
    """

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
        """
        Initialize the baseball simulator with model, scaler and model settings from the config file.

        Args:
            idata file
            scaler file

        """
        self.idata = idata
        self.scaler = scaler

        # Define outcome category mapping (same as used in training)
        self.outcome_labels = outcome_labels
        self.n_categories = len(self.outcome_labels)

        # Define predictor columns
        self.predictor_cols = predictor_cols

        # Identify continuous columns needing scaling vs categorical
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols

        # Default league average rates - can be overridden
        self.league_avg_rates = league_avg_rates

        # --- Pre-calculate mean posterior parameters ---
        logger.info(
            "Pre-calculating mean posterior parameters",
        )  # Optional: for feedback
        try:
            self.mean_intercepts_val = (
                self.idata.posterior["intercepts"]
                .mean(dim=("chain", "draw"))
                .to_numpy()
            )
            self.mean_betas_val = (
                self.idata.posterior["betas"].mean(dim=("chain", "draw")).to_numpy()
            )

            # Get n_predictors from the shape of mean_betas_val for later checks
            # Assuming mean_betas_val shape is (n_predictors, n_categories) or (n_predictors, n_categories-1)
            self.n_model_predictors = self.mean_betas_val.shape[0]

            logger.info(
                "Mean posterior parameters pre-calculated successfully.",
            )  # Optional
        except Exception as e:
            logger.warning(
                f"FATAL ERROR: Could not pre-calculate mean posterior parameters during simulator initialization: {e}",
            )
            logger.info(
                "Make sure 'intercepts' and 'betas' exist in idata.posterior and are correctly formatted.",
            )
            raise
        # --- End of pre-calculation ---

    def set_league_avg_rates(self, rates_dict: dict[str, float]) -> None:
        """
        Update league average rates used in simulation.

        Args:
            rates_dict: Dictionary containing rate names and values

        """
        self.league_avg_rates.update(rates_dict)

    def predict_pa_outcome_probs(self, pa_inputs_dict: dict[str, float]) -> np.ndarray:
        """
        Predicts outcome probabilities for a single plate appearance using MEAN model posterior parameters.

        Args:
            pa_inputs_dict: Dictionary with predictor names as keys and single values.

        Returns:
            np.ndarray: A probability vector (sums to 1) for the PA outcomes. Shape: (n_categories,)

        """
        # 1. Prepare Input Data into correct order
        try:
            input_list = [pa_inputs_dict[col] for col in self.predictor_cols]
            input_array = np.array(input_list).reshape(
                1,
                -1,
            )  # Reshape to 2D array (1 row)
        except KeyError as e:
            logger.warning(f"Error: Missing key {e} in pa_inputs_dict")
            logger.warning(f"Required keys: {self.predictor_cols}")
            raise

        # Use Pandas for easier column selection based on names for scaling
        input_df = pd.DataFrame(input_array, columns=self.predictor_cols)

        # 2. Scale Continuous Features
        continuous_data = input_df[self.continuous_cols].to_numpy()
        try:
            scaled_continuous_data = self.scaler.transform(
                continuous_data,
            )  # Use transform, NOT fit_transform
        except Exception as e:
            logger.warning(f"Error scaling data: {e}")
            logger.warning(f"Input data shape for scaling: {continuous_data.shape}")
            logger.warning(f"Scaler expects {self.scaler.n_features_in_} features.")
            raise

        # 3. Combine Features
        categorical_data = input_df[self.categorical_cols].to_numpy()
        categorical_data = np.asarray(categorical_data)
        # Ensure both arrays are 2D and have compatible dtypes
        if categorical_data.ndim == 1:
            categorical_data = categorical_data.reshape(-1, 1)
        # Convert categorical_data to float if needed for compatibility
        categorical_data = categorical_data.astype(float)
        X_new = np.concatenate([scaled_continuous_data, categorical_data], axis=1)

        n_input_features = X_new.shape[1]  # Number of features in the current input

        if n_input_features != self.n_model_predictors:
            # This check is crucial if the number of predictors could vary or if there's a mismatch
            # between predictor_cols used for input prep and what the model was trained on.
            raise ValueError(
                f"Mismatch in number of features. Input has {n_input_features}, "
                f"but model (from beta shapes) expects {self.n_model_predictors}.",
            )

        # 4. Use PRE-CALCULATED Mean Posterior Parameters
        # We use self.mean_intercepts_val and self.mean_betas_val calculated in __init__
        mean_intercepts = self.mean_intercepts_val
        mean_betas = self.mean_betas_val

        # Check shapes (now against the pre-calculated and stored shapes)
        expected_intercept_shape = (self.n_categories,)
        # The expected_beta_shape should now use self.n_model_predictors which was derived from the loaded betas
        expected_beta_shape = (self.n_model_predictors, self.n_categories)

        if mean_intercepts.shape != expected_intercept_shape:
            logger.warning(
                f"Warning: Mean intercepts shape mismatch. Expected {expected_intercept_shape}, got {mean_intercepts.shape}",
            )
            if mean_intercepts.shape == (self.n_categories - 1,):
                mean_intercepts = np.concatenate([mean_intercepts, [0.0]])
                self.mean_intercepts_val = mean_intercepts  # Optionally update the stored value if you allow this modification
            else:
                raise ValueError(
                    "Cannot resolve intercept shape mismatch with pre-calculated values.",
                )

        if mean_betas.shape != expected_beta_shape:
            logger.warning(
                f"Warning: Mean betas shape mismatch. Expected {expected_beta_shape}, got {mean_betas.shape}",
            )
            if mean_betas.shape == (self.n_model_predictors, self.n_categories - 1):
                ref_betas = np.zeros((self.n_model_predictors, 1))
                mean_betas = np.concatenate([mean_betas, ref_betas], axis=1)
                self.mean_betas_val = mean_betas  # Optionally update the stored value
            else:
                raise ValueError(
                    "Cannot resolve beta shape mismatch with pre-calculated values.",
                )

        # 5. Calculate Linear Predictor (mu)
        mu_mean = mean_intercepts + X_new @ mean_betas  # Shape (1, n_categories)

        # 5. Calculate Linear Predictor (mu)
        mu_mean = mean_intercepts + X_new @ mean_betas  # Shape (1, n_categories)

        # 6. Apply Softmax (manual implementation for stability)
        exp_mu_mean = np.exp(mu_mean - np.max(mu_mean, axis=1, keepdims=True))
        p_vector_mean = exp_mu_mean / np.sum(exp_mu_mean, axis=1, keepdims=True)

        # Ensure probabilities sum roughly to 1
        if not np.isclose(np.sum(p_vector_mean), 1.0):
            logger.warning(
                f"Warning: Probabilities do not sum to 1: {np.sum(p_vector_mean)}",
            )
            # Normalize as fallback
            p_vector_mean = p_vector_mean / np.sum(p_vector_mean)

        return p_vector_mean.flatten()  # Return the single probability vector

    def simulate_single_inning(
        self,
        inning_num: int,
        is_top_inning: bool,
        lineup: list[dict[str, Any]],
        start_batter_idx: int,
        pitcher_inputs: dict[str, Any],
        game_context: dict[str, Any],
    ) -> tuple[int, int, int, int, int]:
        """
        Simulates a single half-inning with realistic base running.

        Args:
            inning_num: The inning number being simulated
            is_top_inning: True if top half of inning (away team batting)
            lineup: List of dictionaries containing batter stats/inputs
            start_batter_idx: Index in lineup to start with
            pitcher_inputs: Dictionary with pitcher stats/inputs
            game_context: Dict containing park factor and team defense ratings

        Returns:
            Tuple of (hits, runs, walks, next_batter_idx)

        """
        outs = 0
        hits = 0
        runs = 0
        walks = 0  # Added walk tracking
        home_runs = 0
        bases = [0, 0, 0]  # 0=empty, 1=runner present
        current_batter_idx = start_batter_idx
        lineup_len = len(lineup)

        # Determine fielding team's defense rating from game_context
        if is_top_inning:  # Away team batting, Home team fielding
            fielding_team_defense_rating = game_context["home_team_defense_rating"]
            is_batter_home = 0
        else:  # Home team batting, Away team fielding
            fielding_team_defense_rating = game_context["away_team_defense_rating"]
            is_batter_home = 1

        inning_context_pa = {
            "park_factor_input": game_context["park_factor_input"],
            "team_defense_oaa_input": fielding_team_defense_rating,
            "is_batter_home": is_batter_home,
        }

        while outs < 3:
            batter_spot_in_lineup = current_batter_idx % lineup_len
            current_batter_inputs = lineup[batter_spot_in_lineup]

            try:
                batter_stand = current_batter_inputs["stand"]
                pitcher_throws = pitcher_inputs["p_throws"]
                is_platoon = (
                    1
                    if (batter_stand == "L" and pitcher_throws == "R")
                    or (batter_stand == "R" and pitcher_throws == "L")
                    else 0
                )
            except KeyError as e:
                logger.warning(
                    f"Warning: Missing 'stand' or 'p_throws' in inputs: {e}. Assuming no platoon advantage.",
                )
                is_platoon = 0

            pa_inputs = {
                **current_batter_inputs,
                **pitcher_inputs,
                **inning_context_pa,
                "is_platoon_adv": is_platoon,
            }
            # Remove non-predictor keys if they exist from batter/pitcher inputs
            pa_inputs = {
                k: v
                for k, v in pa_inputs.items()
                if k in self.predictor_cols or k in ["stand", "p_throws"]
            }

            outcome_probs = self.predict_pa_outcome_probs(pa_inputs)

            possible_outcomes = list(self.outcome_labels.keys())
            simulated_outcome_code = np.random.choice(
                possible_outcomes,
                p=outcome_probs,
            )
            outcome_label = self.outcome_labels[simulated_outcome_code]

            new_bases = list(bases)
            runs_this_pa = 0
            pa_hit = 0
            pa_walk = 0
            pa_hr = 0

            # Store outs *before* this PA is resolved for GIDP check
            outs_before_pa = outs

            if outcome_label == "Strikeout":
                outs += 1
            elif outcome_label == "Walk":
                pa_walk += 1  # Track walk
                # Force runner advancement logic (simplified)
                if bases[0] == 1:
                    if bases[1] == 1:
                        if bases[2] == 1:
                            runs_this_pa += 1
                        new_bases[2] = 1
                    new_bases[1] = 1
                new_bases[0] = 1
            elif outcome_label == "HBP":
                # Force runner advancement logic (same as walk)
                if bases[0] == 1:
                    if bases[1] == 1:
                        if bases[2] == 1:
                            runs_this_pa += 1
                        new_bases[2] = 1
                    new_bases[1] = 1
                new_bases[0] = 1
            elif outcome_label == "Single":
                pa_hit += 1
                # Advance runners (simplified rules + probabilistic 1st->3rd)
                runner_3b_scores = bases[2] == 1
                runner_2b_scores = bases[1] == 1  # Assume scores from 2nd
                runner_1b_to_3rd = False
                runner_1b_to_2nd = False

                if runner_3b_scores:
                    runs_this_pa += 1
                if runner_2b_scores:
                    runs_this_pa += 1

                if bases[0] == 1:
                    if (
                        random.random()
                        < self.league_avg_rates["rate_1st_to_3rd_on_single"]
                    ):
                        runner_1b_to_3rd = True
                    else:
                        runner_1b_to_2nd = True

                # Place runners
                new_bases = [0, 0, 0]
                if runner_1b_to_3rd:
                    new_bases[2] = 1
                elif runner_2b_scores == False and bases[1] == 1:  # noqa: E712
                    new_bases[2] = 1  # R2 holds 3rd if didn't score
                if runner_1b_to_2nd:
                    new_bases[1] = 1
                new_bases[0] = 1  # Batter to 1st

            elif outcome_label == "Double":
                pa_hit += 1
                runner_3b_scores = bases[2] == 1
                runner_2b_scores = bases[1] == 1
                runner_1b_to_3rd = False

                if runner_3b_scores:
                    runs_this_pa += 1
                if runner_2b_scores:
                    runs_this_pa += 1
                if bases[0] == 1:
                    if (
                        random.random()
                        < self.league_avg_rates["rate_score_from_1st_on_double"]
                    ):
                        runs_this_pa += 1
                    else:
                        runner_1b_to_3rd = True

                new_bases = [0, 0, 0]
                new_bases[1] = 1  # Batter to 2nd
                if runner_1b_to_3rd:
                    new_bases[2] = 1

            elif outcome_label == "Triple":
                pa_hit += 1
                runs_this_pa += sum(bases)  # All runners score
                new_bases = [0, 0, 1]  # Batter to 3rd

            elif outcome_label == "HomeRun":
                pa_hit += 1
                pa_hr += 1
                runs_this_pa += 1 + sum(bases)
                new_bases = [0, 0, 0]

            elif outcome_label == "Out_In_Play":
                outs += 1
                # Check GIDP opportunity (runner on 1st, less than 2 outs *before* this PA)
                is_gidp_opportunity = bases[0] == 1 and outs_before_pa < 2
                # Use adjusted rate directly, as bb_type isn't predicted
                adjusted_gidp_rate = self.league_avg_rates.get(
                    "gidp_effective_rate",
                    0.065,
                )  # Use pre-calculated effective rate

                if is_gidp_opportunity and random.random() < adjusted_gidp_rate:
                    if outs < 3:  # Ensure DP doesn't add 4th out
                        outs += 1  # It's a double play
                    # Simplified GIDP: batter out, runner forced at 2nd is out, others hold/advance if forced by other runners
                    runner_3b_holds = bases[2] == 1
                    runner_2b_to_3rd = bases[1] == 1
                    new_bases = [0, 0, 0]  # Batter out, runner from 1st out at 2nd
                    if runner_2b_to_3rd:
                        new_bases[2] = 1  # Runner from 2nd takes 3rd
                    if runner_3b_holds and not runner_2b_to_3rd:
                        new_bases[2] = 1  # Runner from 3rd holds if not pushed

                else:  # Not a GIDP
                    # Batter is out, advance runners if forced (simplified: 1 base)
                    runner_3b_holds = bases[2] == 1
                    runner_2b_to_3rd = bases[1] == 1
                    runner_1b_to_2nd = bases[0] == 1
                    new_bases = [0, 0, 0]  # Batter out
                    if runner_1b_to_2nd:
                        new_bases[1] = 1
                    if runner_2b_to_3rd:
                        new_bases[2] = 1
                    if runner_3b_holds and not runner_2b_to_3rd:
                        new_bases[2] = 1

            # Update inning totals and base state
            runs += runs_this_pa
            hits += pa_hit
            walks += pa_walk
            home_runs += pa_hr
            bases = new_bases

            # Move to next batter for next loop iteration
            current_batter_idx += 1

        # Inning Over
        return hits, runs, walks, home_runs, (current_batter_idx % lineup_len)

    def simulate_first_three_innings(
        self,
        home_lineup: list[dict[str, Any]],
        away_lineup: list[dict[str, Any]],
        home_pitcher_inputs: dict[str, Any],
        away_pitcher_inputs: dict[str, Any],
        game_context: dict[str, Any],
    ) -> dict[str, dict[str, dict[str, int]]]:
        """
        Simulates the first 3 innings of a game.

        Args:
            home_lineup: List of dictionaries with home team batter stats
            away_lineup: List of dictionaries with away team batter stats
            home_pitcher_inputs: Dictionary with home pitcher stats
            away_pitcher_inputs: Dictionary with away pitcher stats
            game_context: Dict with park_factor_input, home_team_defense_rating,
                          away_team_defense_rating

        Returns:
            Dict: Results containing hits, runs, walks per team per inning.
                 Example: {'inning_1': {'away': {'H':1,'R':0,'BB':0}, 'home': {'H':0,'R':0,'BB':1}}, ...}

        """
        results = {}
        away_batter_idx = 0
        home_batter_idx = 0

        for inning in range(1, 4):  # Innings 1, 2, 3
            inning_results = {"away": {}, "home": {}}

            # --- Top of Inning ---
            away_hits, away_runs, away_walks, away_hrs, away_batter_idx_next = (
                self.simulate_single_inning(
                    inning,
                    True,
                    away_lineup,
                    away_batter_idx,
                    home_pitcher_inputs,
                    game_context,
                )
            )
            inning_results["away"] = {
                "H": away_hits,
                "R": away_runs,
                "BB": away_walks,
                "HR": away_hrs,
            }
            away_batter_idx = away_batter_idx_next  # Update for next away inning

            # --- Bottom of Inning ---
            home_hits, home_runs, home_walks, home_hrs, home_batter_idx_next = (
                self.simulate_single_inning(
                    inning,
                    False,
                    home_lineup,
                    home_batter_idx,
                    away_pitcher_inputs,
                    game_context,
                )
            )
            inning_results["home"] = {
                "H": home_hits,
                "R": home_runs,
                "BB": home_walks,
                "HR": home_hrs,
            }
            home_batter_idx = home_batter_idx_next  # Update for next home inning

            results[f"inning_{inning}"] = inning_results

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
        """
        Run multiple simulations of the first three innings.

        Args:
            home_lineup: List of dictionaries with home team batter stats
            away_lineup: List of dictionaries with away team batter stats
            home_pitcher_inputs: Dictionary with home pitcher stats
            away_pitcher_inputs: Dictionary with away pitcher stats
            game_context: Dict with park_factor_input and team defense ratings
            num_sims: Number of simulations to run

        Returns:
            List of simulation results

        """
        all_results = []
        logger.info(f"Running {num_sims} game simulations...")  # noqa: G004
        for _ in tqdm(range(num_sims)):
            sim_result = self.simulate_first_three_innings(
                home_lineup,
                away_lineup,
                home_pitcher_inputs,
                away_pitcher_inputs,
                game_context,
            )
            all_results.append(sim_result)

        return all_results

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
