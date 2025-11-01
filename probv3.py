#!/usr/bin/env python3
import numpy as np
from typing import List, Tuple

# --- V3 HYPERPARAMETERS ---
ALPHA_TV_PENALTY = 0.5   # Total Variation Damping factor
BETA_TRANS_PENALTY = 0.1 # Transition Matrix Penalty factor (Probabilistic Foundation)

class ProbabilityStasisV3:
    """
    V3: Enhanced filter using Total Variation (TV) + Probabilistic Transition Penalty.
    
    This filter penalizes total ruggedness and specifically punishes non-productive 
    transitions between discrete confidence states (H/M/L).
    """

    def __init__(self, alpha_tv: float = ALPHA_TV_PENALTY, beta_trans: float = BETA_TRANS_PENALTY, max_keep: int = 1):
        self.alpha_tv = alpha_tv
        self.beta_trans = beta_trans
        self.max_keep = max_keep

    def _get_state(self, p: float) -> str:
        """Discretizes probability into High, Moderate, or Low confidence states."""
        if p >= 0.85:
            return 'H'
        elif p > 0.55:
            return 'M'
        else:
            return 'L'
    
    def _calculate_transition_penalty(self, probs: np.ndarray) -> float:
        """Calculates the penalty based on state transitions (Probabilistic Foundation)."""
        if len(probs) < 2:
            return 0.0

        total_penalty = 0.0
        
        # Define the penalty for non-productive transitions
        # Penalty is (Current State, Next State): Weight
        PENALTY_MAP = {
            ('H', 'L'): 3.0, # Catastrophic failure
            ('L', 'H'): 1.0, # Sudden, unjustified jump
            ('H', 'M'): 0.5, # Acceptable decrease
        }

        for i in range(len(probs) - 1):
            state_curr = self._get_state(probs[i])
            state_next = self._get_state(probs[i+1])
            transition = (state_curr, state_next)
            
            total_penalty += PENALTY_MAP.get(transition, 0.0)
            
        return total_penalty

    def stasis_score(self, probs: List[float]) -> float:
        """
        Calculates the V3 Stasis Score: Mean - (TV Penalty) - (Transition Penalty).
        """
        p = np.array(probs)
        if len(p) == 0:
            return -np.inf

        mean_val = np.mean(p)
        
        # 1. Total Variation (TV) Penalty
        total_variation = np.sum(np.abs(p[1:] - p[:-1])) if len(p) > 1 else 0.0
        tv_penalty = self.alpha_tv * total_variation
        
        # 2. Transition Matrix Penalty (Probabilistic)
        transition_penalty = self.beta_trans * self._calculate_transition_penalty(p)
        
        # 3. Final Score
        return mean_val - tv_penalty - transition_penalty

    def filter_paths(self, paths: List[Tuple[str, List[float]]]) -> List[Tuple[str, float]]:
        """Calculates scores for all paths and returns the top 'max_keep' paths."""
        scored_paths = []
        for name, probs in paths:
            score = self.stasis_score(probs)
            scored_paths.append((name, score))

        scored_paths.sort(key=lambda x: x[1], reverse=True)
        return scored_paths[:self.max_keep]

if __name__ == '__main__':
    # V3 Self-Test: TV and Transition Penalty in action
    engine = ProbabilityStasisV3(alpha_tv=0.5, beta_trans=0.1, max_keep=1)
    
    # Path X (V2 Winner): Stable/Reliable. Transitions: H->H->H. TV=0.10. D_trans=0.0.
    path_x = ("X: V3 Baseline", [0.92, 0.95, 0.90, 0.92])
    # Expected Score: 0.9225 - 0.5*0.10 - 0.1*0.0 = 0.8725
    
    # Path Y (V1 Jumpy): Catastrophic drop, but overall TV is moderate. 
    # Transitions: H->L (3.0 penalty) -> H (1.0 penalty). TV=0.9. D_trans=4.0.
    path_y = ("Y: Catastrophic Jumps", [0.95, 0.40, 0.90, 0.85])
    # Expected Score: 0.7750 - 0.5*0.90 - 0.1*4.0 = 0.775 - 0.45 - 0.40 = -0.075
    
    # Path Z (TV Challenge): Smoothly decreasing TV=0.30. Transitions: H->M->M. D_trans=0.5.
    path_z = ("Z: Smooth Decay", [0.90, 0.80, 0.70, 0.60])
    # Expected Score: 0.7500 - 0.5*0.30 - 0.1*0.5 = 0.75 - 0.15 - 0.05 = 0.5500

    results = engine.filter_paths([path_x, path_y, path_z])
    
    print("--- Probability Stasis V3 (TV + Transition Penalty) Test ---")
    print(f"Hyperparameters: TV={engine.alpha_tv}, Trans={engine.beta_trans}")
    print(f"Top Path:")
    print(f"  {results[0][0]}: {results[0][1]:.4f}")
    assert results[0][0] == "X: V3 Baseline"
    print("Test Passed: Baseline path correctly wins, Catastrophic Jumps path severely penalized.")
