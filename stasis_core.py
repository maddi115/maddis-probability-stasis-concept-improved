#!/usr/bin/env python3
import numpy as np
from typing import List, Tuple

class ProbabilityStasis:
    """
    Enhanced Probabilistic Language Model filter based on Mean Confidence and 
    a specialized Max Downward Jitter Penalty (mu).

    The Stasis Score prioritizes reasoning paths that maintain conviction, 
    severely penalizing the largest single drop in confidence (loss of faith).

    Formula: Score = Mean(P) - mu * Max_Downward_Jitter(P)
    """

    def __init__(self, mu_jitter_penalty: float = 1.5, max_keep: int = 1):
        """
        Initializes the Stasis filter.

        :param mu_jitter_penalty: Penalty factor for the Max Downward Jitter (mu). 
                                  Higher mu means greater penalty for confidence drops.
        :param max_keep: The maximum number of top-scoring paths to return.
        """
        self.mu_jitter_penalty = mu_jitter_penalty
        self.max_keep = max_keep

    def stasis_score(self, probs: List[float]) -> float:
        """
        Calculates the Stasis Score for a single path.
        """
        p = np.array(probs)
        if len(p) == 0:
            return -np.inf

        # 1. Mean (Confidence)
        mean_val = np.mean(p)

        # 2. Max Downward Jitter (The Instability Penalty)
        # Calculate P[i-1] - P[i] and find the largest drop (positive difference)
        if len(p) <= 1:
            max_drop = 0.0
        else:
            drops = p[:-1] - p[1:]
            # Only consider positive drops (loss of confidence)
            max_drop = np.max(np.concatenate(([0.0], drops[drops > 0])))
        
        # 3. Apply Jitter Penalty
        jitter_penalty = self.mu_jitter_penalty * max_drop
        
        # 4. Final Score
        return mean_val - jitter_penalty

    def filter_paths(self, paths: List[Tuple[str, List[float]]]) -> List[Tuple[str, float]]:
        """
        Calculates scores for all paths and returns the top 'max_keep' paths.

        :param paths: List of tuples (path_name, [probabilities]).
        :return: List of tuples (path_name, stasis_score), sorted descending.
        """
        scored_paths = []
        for name, probs in paths:
            score = self.stasis_score(probs)
            scored_paths.append((name, score))

        # Sort by score (highest first)
        scored_paths.sort(key=lambda x: x[1], reverse=True)
        
        return scored_paths[:self.max_keep]

if __name__ == '__main__':
    # Simple self-test of the final class
    engine = ProbabilityStasis(mu_jitter_penalty=1.5, max_keep=2)
    test_paths = [
        ("A: Stable High", [0.95, 0.94, 0.95, 0.94]), # Max Drop 0.01
        ("B: Jumpy Low", [0.80, 0.50, 0.80, 0.50]),    # Max Drop 0.30
        ("C: Very High Mean, Jumpy", [0.99, 0.80, 0.99, 0.80]), # Max Drop 0.19
    ]
    
    # Expected Scores:
    # A: 0.945 - 1.5*0.01 = 0.930
    # B: 0.675 - 1.5*0.30 = 0.225
    # C: 0.905 - 1.5*0.19 = 0.620
    
    results = engine.filter_paths(test_paths)
    print("--- Probability Stasis Final Module Test ---")
    print(f"Top 2 Paths (mu={engine.mu_jitter_penalty}):")
    for name, score in results:
        print(f"  {name}: {score:.3f}")
    # A should beat C, and C should beat B
    assert results[0][0] == "A: Stable High"
    assert results[1][0] == "C: Very High Mean, Jumpy"
    print("Test Passed: Stable High path correctly wins over higher mean, jumpier path.")

