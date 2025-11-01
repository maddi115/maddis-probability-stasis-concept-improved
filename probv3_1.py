#!/usr/bin/env python3
"""
Probability Stasis Filter V3.1 (Dynamic Penalties)
Implements adaptive penalty weights based on the path's starting confidence (P0).
"""
import numpy as np
from typing import List, Tuple
from tabulate import tabulate
from probv3 import ProbabilityStasisV3 # Inherit the V3 base logic

# --- BASE HYPERPARAMETERS (Same as V3) ---
ALPHA_TV_PENALTY = 0.5
BETA_TRANS_PENALTY = 0.1

class ProbabilityStasisV3_1(ProbabilityStasisV3):
    """
    V3.1: Implements dynamic penalties that scale based on the initial probability P0.
    A path starting near 0 or 1 is penalized more severely for instability.
    """

    def _get_dynamic_weights(self, p0: float) -> Tuple[float, float]:
        """Calculates dynamic alpha and beta based on the starting probability P0."""
        
        # Calculate the confidence multiplier: 1.0 + |P0 - 0.5|
        # This multiplier is minimal (1.0) at P0=0.5 and maximal (1.5) at P0=0 or P0=1.
        multiplier = 1.0 + abs(p0 - 0.5)
        
        alpha_dyn = self.alpha_tv * multiplier
        beta_dyn = self.beta_trans * multiplier
        
        return alpha_dyn, beta_dyn

    def stasis_score(self, probs: List[float]) -> float:
        """
        Calculates the V3.1 Stasis Score: Mean - (Dynamic TV) - (Dynamic Trans).
        """
        p = np.array(probs)
        if len(p) == 0:
            return -np.inf

        mean_val = np.mean(p)
        p0 = p[0] # The starting probability dictates the dynamic weight
        
        alpha_dyn, beta_dyn = self._get_dynamic_weights(p0)
        
        # V3 Components
        total_variation = np.sum(np.abs(p[1:] - p[:-1])) if len(p) > 1 else 0.0
        
        # Dynamic Penalty Calculation
        tv_penalty = alpha_dyn * total_variation
        transition_penalty = beta_dyn * self._calculate_transition_penalty(p)

        total_penalty = tv_penalty + transition_penalty
        
        return mean_val - total_penalty

if __name__ == '__main__':
    # Define the same simulated paths for a direct comparison
    SIMULATED_PATHS = [
        ("Path A: High Volatility", [0.90, 0.30, 0.85, 0.90], "P0=0.90"),
        ("Path B: Stable & Moderate", [0.70, 0.75, 0.70, 0.75], "P0=0.70"),
        ("Path C: Low Mean Jitter", [0.60, 0.55, 0.60, 0.55], "P0=0.60"),
        ("Path D: Perfect Stability", [0.95, 0.95, 0.95, 0.95], "P0=0.95"),
        ("Path E: High Jitter", [0.95, 0.75, 0.95, 0.75], "P0=0.95"),
    ]

    engine_v3_1 = ProbabilityStasisV3_1(alpha_tv=ALPHA_TV_PENALTY, beta_trans=BETA_TRANS_PENALTY, max_keep=5)
    
    analysis_data = []
    
    print("--- 🔬 V3.1 Dynamic Penalty Analysis (Static V3 Scores are in parentheses) ---")
    headers = ["Path", "P0", "V3.1 Score", "Static V3 Score", "TV Penalty", "Trans Penalty", "Mean"]
    
    for name, probs, _ in SIMULATED_PATHS:
        score_v3_1 = engine_v3_1.stasis_score(probs)
        
        p = np.array(probs)
        p0 = p[0]
        mean_val = np.mean(p)

        # Calculate V3.1 Penalties for display
        alpha_dyn, beta_dyn = engine_v3_1._get_dynamic_weights(p0)
        total_variation = np.sum(np.abs(p[1:] - p[:-1])) if len(p) > 1 else 0.0
        
        tv_penalty = alpha_dyn * total_variation
        transition_penalty = beta_dyn * engine_v3_1._calculate_transition_penalty(p)
        
        # Calculate Static V3 Score for direct comparison
        # (Using the base V3.1 engine with base weights to calculate the penalty for simplicity)
        engine_v3_base = ProbabilityStasisV3(alpha_tv=ALPHA_TV_PENALTY, beta_trans=BETA_TRANS_PENALTY)
        score_v3_static = engine_v3_base.stasis_score(probs)


        analysis_data.append([
            name,
            f"{p0:.2f}",
            f"{score_v3_1:.4f}",
            f"{score_v3_static:.4f}",
            f"{tv_penalty:.4f} (Dyn)",
            f"{transition_penalty:.4f} (Dyn)",
            f"{mean_val:.4f}"
        ])
        
    print(tabulate(analysis_data, headers=headers, tablefmt="fancy_grid"))
    
    # Print V3.1 Ranking
    scored_paths_v3_1 = [(item[0], float(item[2])) for item in analysis_data]
    scored_paths_v3_1.sort(key=lambda x: x[1], reverse=True)
    print("\n--- 🏆 V3.1 (Dynamic) RANKING ---")
    print(tabulate(scored_paths_v3_1, headers=["Path Name", "V3.1 Stasis Score"], tablefmt="fancy_grid"))

