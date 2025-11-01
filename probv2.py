#!/usr/bin/env python3
import numpy as np
from typing import List, Tuple

class ProbabilityStasisV2:
    """
    V2: Enhanced filter using Total Variation (TV) Damping.
    
    This filter penalizes the TOTAL change in confidence across all steps, 
    measuring the sequence's 'ruggedness.'
    
    Formula: Score = Mean(P) - alpha * Total_Variation(P)
    """

    def __init__(self, alpha_tv_penalty: float = 0.5, max_keep: int = 1):
        """
        Initializes the V2 Stasis filter.

        :param alpha_tv_penalty: Penalty factor for the Total Variation (alpha). 
                                  Higher alpha means greater penalty for overall ruggedness.
        """
        self.alpha_tv_penalty = alpha_tv_penalty
        self.max_keep = max_keep

    def stasis_score(self, probs: List[float]) -> float:
        """
        Calculates the Stasis Score for a single path using Total Variation.
        """
        p = np.array(probs)
        if len(p) == 0:
            return -np.inf

        # 1. Mean (Confidence)
        mean_val = np.mean(p)

        # 2. Total Variation (TV) - Sum of absolute differences between steps
        # TV = sum(|P[i+1] - P[i]|)
        if len(p) <= 1:
            total_variation = 0.0
        else:
            total_variation = np.sum(np.abs(p[1:] - p[:-1]))
        
        # 3. Apply Total Variation Damping
        tv_penalty = self.alpha_tv_penalty * total_variation
        
        # 4. Final Score
        return mean_val - tv_penalty

    def filter_paths(self, paths: List[Tuple[str, List[float]]]) -> List[Tuple[str, float]]:
        """
        Calculates scores for all paths and returns the top 'max_keep' paths.
        """
        scored_paths = []
        # Correctly unpacks only two values (name and probs)
        for name, probs in paths: 
            score = self.stasis_score(probs)
            scored_paths.append((name, score))

        scored_paths.sort(key=lambda x: x[1], reverse=True)
        
        return scored_paths[:self.max_keep]

if __name__ == '__main__':
    # Simple self-test for the Total Variation logic (Paths corrected to 2 items)
    engine = ProbabilityStasisV2(alpha_tv_penalty=0.5, max_keep=1)
    
    # Corrected test paths: (Name, [Probabilities])
    path_x = ("X: Smooth Drop", [0.85, 0.75, 0.65])
    path_y = ("Y: Very Jumpy", [0.90, 0.50, 0.70])
    path_z = ("Z: Perfectly Stable", [0.70, 0.70, 0.70])

    results = engine.filter_paths([path_x, path_y, path_z])
    
    print("--- Probability Stasis V2 (Total Variation Damping) Test ---")
    print(f"Top Path (alpha={engine.alpha_tv_penalty}):")
    print(f"  {results[0][0]}: {results[0][1]:.3f}")
    assert results[0][0] == "Z: Perfectly Stable"
    print("Test Passed: Perfectly Stable path correctly wins by minimizing Total Variation.")
