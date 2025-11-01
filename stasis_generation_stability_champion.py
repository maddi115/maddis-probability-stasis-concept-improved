#!/usr/bin/env python3
"""
Demonstrates the IMPROVED Probability Stasis filtering where Extreme Stability (low Max Drop)
wins over a marginally higher Mean (Path X).
"""

import numpy as np
from tabulate import tabulate
from typing import List, Tuple

# --- HYPERPARAMETER (Same as before) ---
MU_JITTER_PENALTY = 1.5 

# Define the scoring function (remains the same)
ReasoningPath = Tuple[str, List[float], List[str]]

def improved_stasis_score_max_jitter(probs: List[float], mu_val: float) -> Tuple[float, float, float]:
    p = np.array(probs)
    mean_val = np.mean(p)
    
    if len(p) <= 1:
        max_drop = 0.0
    else:
        drops = p[:-1] - p[1:]
        max_drop = np.max(np.concatenate(([0.0], drops[drops > 0])))
    
    jitter_penalty = mu_val * max_drop
    score = mean_val - jitter_penalty
    return score, mean_val, max_drop, jitter_penalty

def run_stasis_generation_max_jitter(query: str, paths: List[ReasoningPath], mu_val: float):
    """Runs the final improved Max Jitter simulation."""
    print(f"--- 💡 Query: {query} ---")
    print(f"--- 🌟 FINAL TEST: Stability vs. Mean (μ={mu_val}) ---")

    analysis_results = []
    
    for name, probs, text_steps in paths:
        score, mean_val, max_drop, penalty = improved_stasis_score_max_jitter(probs, mu_val)
        
        analysis_results.append({
            "name": name,
            "score": score,
            "mean": f"{mean_val:.4f}",
            "max_drop": f"{max_drop:.4f}",
            "penalty": f"{penalty:.4f}",
            "text_steps": text_steps
        })

    sorted_results = sorted(analysis_results, key=lambda x: x['score'], reverse=True)

    # Display Analysis
    table_data = []
    for rank, data in enumerate(sorted_results):
        table_data.append([
            rank + 1,
            data['name'],
            data['mean'],
            data['max_drop'],
            data['penalty'],
            f"{data['score']:.4f}"
        ])
    
    headers = ["Rank", "Path", "Mean", "Max Drop", "Penalty (μ*Drop)", "New StasisScore"]
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

    # Select and Generate Final Text
    winning_path_name = sorted_results[0]['name']
    winning_path_data = next((p for p in paths if p[0] == winning_path_name), None)
    winning_steps = winning_path_data[2]
    final_answer = winning_steps[-1] 

    print(f"\n[Step 2: Final Answer Selection]")
    print(f"🏆 Winning Path (Highest New Stasis Score): {winning_path_name}")
    print("\n=========================================================")
    print(f"FINAL STASIS-APPROVED GENERATION:")
    print(f"{final_answer}")
    print("=========================================================\n")


if __name__ == '__main__':
    # --- SIMULATED DATA ---
    QUERY_EXAMPLE = "Which reasoning path is the most reliable?"
    
    # Path X: High Mean / Moderate Drop (Competitor)
    PATH_X = (
        "Path X: High Mean (0.9650) / Moderate Drop (0.03)",
        [0.98, 0.95, 0.96, 0.97], 
        ["This path has the highest average confidence but a slight drop.","Final Answer: High Mean, Slight Jitter."]
    )

    # Path Y: High Fluctuation (Rejected)
    PATH_Y = (
        "Path Y: High Fluctuation (0.7050) / Huge Drop (0.59)",
        [0.55, 0.99, 0.40, 0.88], 
        ["This path is too jumpy to be reliable.","Final Answer: Too Volatile."]
    )

    # Path Z'' (NEW CHAMPION): Low Drop / Sufficient Mean
    PATH_Z_DOUBLE_PRIME = (
        "Path Z'': Stability Champion (0.9450) / Tiny Drop (0.01)",
        [0.95, 0.94, 0.95, 0.94], # Mean 0.9450
        ["This path maintains near-perfect conviction throughout.","Final Answer: Optimal Stability and Reliability."]
    )
    
    simulated_paths = [PATH_X, PATH_Y, PATH_Z_DOUBLE_PRIME]

    # Run the final simulation
    run_stasis_generation_max_jitter(QUERY_EXAMPLE, simulated_paths, MU_JITTER_PENALTY)
