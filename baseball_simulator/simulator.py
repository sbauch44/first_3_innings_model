import logging
import random
from typing import Any

import numpy as np
import polars as pl
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

    def predict_pa_outcome_probs(
        self, pa_inputs_dict: dict[str, float] | list[dict[str, float]]
    ) -> np.ndarray:
        """
        Predicts outcome probabilities for single or multiple plate appearances using MEAN model posterior parameters.

        Args:
            pa_inputs_dict: Dictionary with predictor names as keys and single values,
                        OR list of such dictionaries for batch prediction.

        Returns:
            np.ndarray: A probability vector (sums to 1) for the PA outcomes.
                    Shape: (n_categories,) for single input or (n_samples, n_categories) for batch.
        """
        # Handle both single dict and list of dicts
        if isinstance(pa_inputs_dict, dict):
            pa_inputs_list = [pa_inputs_dict]
            is_single = True
        else:
            pa_inputs_list = pa_inputs_dict
            is_single = False

        # 1. Prepare Input Data into correct order (vectorized)
        try:
            input_arrays = []
            for pa_dict in pa_inputs_list:
                input_list = [pa_dict[col] for col in self.predictor_cols]
                input_arrays.append(input_list)
            input_array = np.array(input_arrays)  # Shape: (n_samples, n_features)
        except KeyError as e:
            logger.warning(f"Error: Missing key {e} in pa_inputs_dict")
            logger.warning(f"Required keys: {self.predictor_cols}")
            raise

        # Direct numpy indexing instead of DataFrame creation
        col_to_idx = {col: idx for idx, col in enumerate(self.predictor_cols)}
        continuous_indices = [col_to_idx[col] for col in self.continuous_cols]
        categorical_indices = [col_to_idx[col] for col in self.categorical_cols]

        # Extract data using direct indexing
        continuous_data = input_array[:, continuous_indices]
        categorical_data = input_array[:, categorical_indices]

        try:
            scaled_continuous_data = self.scaler.transform(continuous_data)
        except Exception as e:
            logger.error(f"Error scaling data: {e}")
            logger.error(f"Input data shape for scaling: {continuous_data.shape}")
            logger.error(
                f"Scaler expects {getattr(self.scaler, 'n_features_in_', 'unknown')} features."
            )
            raise

        # 3. Combine Features
        categorical_data = categorical_data.astype(float)
        X_new = np.concatenate([scaled_continuous_data, categorical_data], axis=1)

        n_input_features = X_new.shape[1]
        if n_input_features != self.n_model_predictors:
            raise ValueError(
                f"Mismatch in number of features. Input has {n_input_features}, "
                f"but model (from beta shapes) expects {self.n_model_predictors}."
            )

        # 4. Use PRE-CALCULATED Mean Posterior Parameters
        mean_intercepts = self.mean_intercepts_val
        mean_betas = self.mean_betas_val

        # Shape validation and correction (same as before)
        expected_intercept_shape = (self.n_categories,)
        expected_beta_shape = (self.n_model_predictors, self.n_categories)

        if mean_intercepts.shape != expected_intercept_shape:
            logger.warning(
                f"Warning: Mean intercepts shape mismatch. Expected {expected_intercept_shape}, got {mean_intercepts.shape}"
            )
            if mean_intercepts.shape == (self.n_categories - 1,):
                mean_intercepts = np.concatenate([mean_intercepts, [0.0]])
                self.mean_intercepts_val = mean_intercepts
            else:
                raise ValueError(
                    "Cannot resolve intercept shape mismatch with pre-calculated values."
                )

        if mean_betas.shape != expected_beta_shape:
            logger.warning(
                f"Warning: Mean betas shape mismatch. Expected {expected_beta_shape}, got {mean_betas.shape}"
            )
            if mean_betas.shape == (self.n_model_predictors, self.n_categories - 1):
                ref_betas = np.zeros((self.n_model_predictors, 1))
                mean_betas = np.concatenate([mean_betas, ref_betas], axis=1)
                self.mean_betas_val = mean_betas
            else:
                raise ValueError(
                    "Cannot resolve beta shape mismatch with pre-calculated values."
                )

        # 5. Calculate Linear Predictor (mu) - now handles batch
        mu_mean = (
            mean_intercepts + X_new @ mean_betas
        )  # Shape (n_samples, n_categories)

        # 6. Apply Softmax (manual implementation for stability)
        exp_mu_mean = np.exp(mu_mean - np.max(mu_mean, axis=1, keepdims=True))
        p_vector_mean = exp_mu_mean / np.sum(exp_mu_mean, axis=1, keepdims=True)

        # Ensure probabilities sum roughly to 1
        prob_sums = np.sum(p_vector_mean, axis=1)
        if not np.allclose(prob_sums, 1.0):
            logger.warning(f"Warning: Some probabilities do not sum to 1: {prob_sums}")
            # Normalize as fallback
            p_vector_mean = p_vector_mean / prob_sums.reshape(-1, 1)

        # Return appropriate shape based on input
        if is_single:
            return p_vector_mean.flatten()  # Return single probability vector
        else:
            return p_vector_mean  # Return batch of probability vectors

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
        Optimized version that batches predictions.
        """
        outs = 0
        hits = 0
        runs = 0
        walks = 0
        home_runs = 0
        bases = [0, 0, 0]
        current_batter_idx = start_batter_idx
        lineup_len = len(lineup)

        # Determine fielding team's defense rating from game_context
        if is_top_inning:
            fielding_team_defense_rating = game_context["home_team_defense_rating"]
            is_batter_home = 0
        else:
            fielding_team_defense_rating = game_context["away_team_defense_rating"]
            is_batter_home = 1

        inning_context_pa = {
            "park_factor_input": game_context["park_factor_input"],
            "team_defense_oaa_input": fielding_team_defense_rating,
            "is_batter_home": is_batter_home,
        }

        # Pre-batch plate appearance inputs for the inning
        # Estimate max PAs per inning (conservative: 3 outs * 5 batters per out = 15)
        max_pas = 15
        pa_inputs_batch = []
        batter_indices_batch = []

        temp_batter_idx = current_batter_idx
        for _ in range(max_pas):
            batter_spot_in_lineup = temp_batter_idx % lineup_len
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
                    f"Warning: Missing 'stand' or 'p_throws' in inputs: {e}. Assuming no platoon advantage."
                )
                is_platoon = 0

            pa_inputs = {
                **current_batter_inputs,
                **pitcher_inputs,
                **inning_context_pa,
                "is_platoon_adv": is_platoon,
            }
            # Remove non-predictor keys
            pa_inputs = {
                k: v
                for k, v in pa_inputs.items()
                if k in self.predictor_cols or k in ["stand", "p_throws"]
            }

            pa_inputs_batch.append(pa_inputs)
            batter_indices_batch.append(temp_batter_idx)
            temp_batter_idx += 1

        # Get all outcome probabilities at once
        all_outcome_probs = self.predict_pa_outcome_probs(pa_inputs_batch)

        # Pre-generate random numbers for outcome selection
        random_values = np.random.random(max_pas)

        # Simulate the inning using pre-calculated probabilities
        pa_count = 0
        possible_outcomes = list(self.outcome_labels.keys())

        while outs < 3 and pa_count < max_pas:
            # Use pre-calculated probabilities
            outcome_probs = all_outcome_probs[pa_count]

            # Convert random value to outcome using cumulative probabilities
            cumsum_probs = np.cumsum(outcome_probs)
            outcome_idx = np.searchsorted(cumsum_probs, random_values[pa_count])
            outcome_idx = min(outcome_idx, len(possible_outcomes) - 1)  # Safety check

            simulated_outcome_code = possible_outcomes[outcome_idx]
            outcome_label = self.outcome_labels[simulated_outcome_code]

            # Update current batter index from pre-calculated batch
            current_batter_idx = batter_indices_batch[pa_count]

            # Rest of the outcome processing logic remains the same
            new_bases = list(bases)
            runs_this_pa = 0
            pa_hit = 0
            pa_walk = 0
            pa_hr = 0
            outs_before_pa = outs

            if outcome_label == "Strikeout":
                outs += 1
            elif outcome_label == "Walk":
                pa_walk += 1
                if bases[0] == 1:
                    if bases[1] == 1:
                        if bases[2] == 1:
                            runs_this_pa += 1
                        new_bases[2] = 1
                    new_bases[1] = 1
                new_bases[0] = 1
            elif outcome_label == "HBP":
                if bases[0] == 1:
                    if bases[1] == 1:
                        if bases[2] == 1:
                            runs_this_pa += 1
                        new_bases[2] = 1
                    new_bases[1] = 1
                new_bases[0] = 1
            elif outcome_label == "Single":
                pa_hit += 1
                runner_3b_scores = bases[2] == 1
                runner_2b_scores = bases[1] == 1
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

                new_bases = [0, 0, 0]
                if runner_1b_to_3rd:
                    new_bases[2] = 1
                elif runner_2b_scores == False and bases[1] == 1:
                    new_bases[2] = 1
                if runner_1b_to_2nd:
                    new_bases[1] = 1
                new_bases[0] = 1

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

                new_bases = [0, 0, 1]  # Batter to 2nd
                if runner_1b_to_3rd:
                    new_bases[2] = 1

            elif outcome_label == "Triple":
                pa_hit += 1
                runs_this_pa += sum(bases)
                new_bases = [0, 0, 1]

            elif outcome_label == "HomeRun":
                pa_hit += 1
                pa_hr += 1
                runs_this_pa += 1 + sum(bases)
                new_bases = [0, 0, 0]

            elif outcome_label == "Out_In_Play":
                outs += 1
                is_gidp_opportunity = bases[0] == 1 and outs_before_pa < 2
                adjusted_gidp_rate = self.league_avg_rates.get(
                    "gidp_effective_rate", 0.065
                )

                if is_gidp_opportunity and random.random() < adjusted_gidp_rate:
                    if outs < 3:
                        outs += 1
                    runner_3b_holds = bases[2] == 1
                    runner_2b_to_3rd = bases[1] == 1
                    new_bases = [0, 0, 0]
                    if runner_2b_to_3rd:
                        new_bases[2] = 1
                    if runner_3b_holds and not runner_2b_to_3rd:
                        new_bases[2] = 1
                else:
                    runner_3b_holds = bases[2] == 1
                    runner_2b_to_3rd = bases[1] == 1
                    runner_1b_to_2nd = bases[0] == 1
                    new_bases = [0, 0, 0]
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

            pa_count += 1

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
        num_sims: int = 10_000,
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

    def convert_probabilities_to_dataframe(self, prob_dict: dict) -> pl.DataFrame:
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
            return pl.DataFrame(data_for_df)
        # Return empty DataFrame with correct columns if no data
        return pl.DataFrame(schema=["inning", "team", "stat", "number", "probability"])
