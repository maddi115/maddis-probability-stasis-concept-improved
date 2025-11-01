#!/usr/bin/env python3
"""
Demonstrates Probability Stasis filtering on a simulated LLM query.
"""

from stasis_core import ProbabilityStasis
from typing import List, Tuple, Dict
import numpy as np
from tabulate import tabulate # Added tabulate for clearer output

# Define the structure of a reasoning path
# Tuple is (Name, [Probabilities], [Generated Text Steps])
ReasoningPath = Tuple[str, List[float], List[str]]

def run_stasis_generation(query: str, paths: List[ReasoningPath], lambda_instability: float = 2.0, max_keep: int = 1):
    """
    Simulates LLM path generation, filters paths using Stasis, and selects the final answer.
    """
    print(f"--- 💡 Query: {query} ---")
    
    # 1. Instantiate the Stasis Engine
    stasis_engine = ProbabilityStasis(lambda_instability=lambda_instability, max_keep=max_keep)
    
    # Extract only the name and probabilities for filtering
    paths_for_filtering = [(name, probs) for name, probs, _ in paths]
    
    # 2. Filter Paths
    print("\n[Step 1: Stasis Filtering]")
    scored_paths = stasis_engine.filter_paths(paths_for_filtering)
    
    analysis_results = []
    
    for rank, (name, score) in enumerate(scored_paths):
        # Find the original path data
        original_path = next((p for p in paths if p[0] == name), None)
        
        if original_path:
            probs = np.array(original_path[1])
            # Calculate components for display
            mean_val = np.mean(probs)
            var_val = np.var(probs)
            range_val = np.max(probs) - np.min(probs)
            
            analysis_results.append([
                rank + 1,
                name,
                f"{mean_val:.4f}",
                f"{var_val:.4f}",
                f"{range_val:.4f}",
                f"{score:.4f}"
            ])

    # Display Analysis using a table
    headers = ["Rank", "Path", "Mean", "Variance", "Range", "StasisScore"]
    print(f"Lambda Instability (λ): {lambda_instability}")
    print(tabulate(analysis_results, headers=headers, tablefmt="fancy_grid"))

    # 3. Select and Generate Final Text
    if not scored_paths:
        print("\n[Step 2: Final Answer] 🚫 No paths survived the Stasis filter.")
        return

    winning_path_name = scored_paths[0][0]
    winning_path_data = next((p for p in paths if p[0] == winning_path_name), None)
    winning_steps = winning_path_data[2]
    final_answer = winning_steps[-1] # Assume the last step contains the final answer

    print(f"\n[Step 2: Final Answer Selection]")
    print(f"🏆 Winning Path (Highest Stasis Score): {winning_path_name}")
    print(f"Path's Reasoning Steps (Last Step is Answer): {winning_steps}")
    print("\n=========================================================")
    print(f"FINAL STASIS-APPROVED GENERATION:")
    print(f"{final_answer}")
    print("=========================================================\n")


if __name__ == '__main__':
    # --- SIMULATED DATA FOR AUSTRALIA CAPITAL ---
    QUERY_EXAMPLE = "What is the capital of Australia and why was that city chosen?"
    
    # Path X: High Stasis (Correct) - High confidence, stable reasoning.
    PATH_X = (
        "Path X: High Stasis (Correct)",
        [0.98, 0.95, 0.96, 0.97], # Very stable, high mean
        [
            "The capital of Australia is Canberra.",
            "It was chosen as a compromise between the two rival cities.",
            "Sydney and Melbourne fiercely competed to be the national capital.",
            "Final Answer: Canberra is the capital. It was specifically selected as a geographically neutral compromise between the two largest, rival cities, Sydney and Melbourne, in 1908."
        ]
    )

    # Path Y: High Fluctuation (Jumpy Correct) - Confused at first, high variance, lands on correct answer.
    PATH_Y = (
        "Path Y: High Fluctuation (Jumpy Correct)",
        [0.55, 0.99, 0.40, 0.88], # Low check, spike, low check, final confidence
        [
            "The capital is either Sydney or Melbourne (low confidence check).",
            "Let's check the historical documents: The capital is actually Canberra, established for the federation.",
            "The selection process was fraught with political arguments over the site.",
            "Final Answer: Canberra is the capital, chosen because it satisfied both Sydney and Melbourne as a non-partisan location."
        ]
    )

    # Path Z: Low Stasis (Stable Wrong) - Stable, but wrong facts result in low stability/medium mean.
    PATH_Z = (
        "Path Z: Stable Wrong (Incorrect)",
        [0.70, 0.75, 0.70, 0.75], # Moderate, stable confidence (overconfident error)
        [
            "The capital of Australia is Sydney.",
            "Sydney is the oldest and most prominent city, so it was the obvious choice.",
            "The location ensures economic power is centered there.",
            "Final Answer: Sydney is the capital of Australia. It was chosen for its early economic and political importance."
        ]
    )
    
    simulated_paths = [PATH_X, PATH_Y, PATH_Z]

    # Run the simulation
    run_stasis_generation(QUERY_EXAMPLE, simulated_paths, lambda_instability=2.0)
