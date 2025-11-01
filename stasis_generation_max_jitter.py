#!/usr/bin/env python3
"""
Demonstrates the IMPROVED Probability Stasis filtering with Max Downward Jitter Penalty (mu).
"""

import numpy as np
from tabulate import tabulate
from typing import List, Tuple

# --- NEW HYPERPARAMETER ---
MU_JITTER_PENALTY = 1.5 # Penalty factor for the Max Downward Jitter

# Define the structure of a reasoning path
ReasoningPath = Tuple[str, List[float], List[str]]

def improved_stasis_score_max_jitter(probs: List[float], mu_val: float) -> Tuple[float, float, float]:
    """Calculates the IMPROVED Stasis Score using Max Downward Jitter."""
    p = np.array(probs)
    
    # 1. Mean (Confidence)
    mean_val = np.mean(p)
    
    # 2. Max Downward Jitter (Loss of Faith)
    # Calculate difference between consecutive steps: P[i-1] - P[i]
    # We only care about positive differences (a drop in confidence).
    if len(p) <= 1:
        max_drop = 0.0
    else:
        drops = p[:-1] - p[1:]
        max_drop = np.max(np.concatenate(([0.0], drops[drops > 0]))) # Max of all drops, or 0
    
    # 3. Apply Penalty
    jitter_penalty = mu_val * max_drop
    
    # NEW FORMULA: Score = Mean - Max_Jitter_Penalty
    score = mean_val - jitter_penalty
    return score, mean_val, max_drop, jitter_penalty

def run_stasis_generation_max_jitter(query: str, paths: List[ReasoningPath], mu_val: float):
    """Runs the improved Max Jitter simulation."""
    print(f"--- 💡 Query: {query} ---")
    print(f"--- 🌟 IMPROVED Stasis Filter (Max Jitter Penalty, μ={mu_val}) ---")

    # 1. Score all paths
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

    # 2. Sort by new Stasis Score (Descending)
    sorted_results = sorted(analysis_results, key=lambda x: x['score'], reverse=True)

    # Display Analysis using a table
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

    # 3. Select and Generate Final Text
    winning_path_name = sorted_results[0]['name']
    winning_path_data = next((p for p in paths if p[0] == winning_path_name), None)
    winning_steps = winning_path_data[2]
    final_answer = winning_steps[-1] 

    print(f"\n[Step 2: Final Answer Selection]")
    print(f"🏆 Winning Path (Highest New Stasis Score): {winning_path_name}")
    print("\n=========================================================")
    print(f"FINAL IMPROVED STASIS-APPROVED GENERATION:")
    print(f"{final_answer}")
    print("=========================================================\n")


if __name__ == '__main__':
    # --- SIMULATED DATA FOR AUSTRALIA CAPITAL (Same Data) ---
    QUERY_EXAMPLE = "What is the capital of Australia and why was that city chosen?"
    
    # Path X: High Stasis (Stable, high)
    PATH_X = (
        "Path X: High Stasis (Correct)",
        [0.98, 0.95, 0.96, 0.97],
        ["The capital is Canberra.", "...","Final Answer: Canberra, stable reasoning."]
    )

    # Path Y: High Fluctuation (Jumpy Correct) - Max drop is 0.99 -> 0.40 = 0.59
    PATH_Y = (
        "Path Y: High Fluctuation (Jumpy Correct)",
        [0.55, 0.99, 0.40, 0.88], 
        ["...","...","...","Final Answer: Canberra, jumpy reasoning that ended high."]
    )

    # Path Z: Low Stasis (Stable Wrong) - Max drop is 0.75 -> 0.70 = 0.05
    PATH_Z = (
        "Path Z: Stable Wrong (Incorrect)",
        [0.70, 0.75, 0.70, 0.75],
        ["The capital is Sydney.","...","...","Final Answer: Sydney, stable reasoning."]
    )
    
    simulated_paths = [PATH_X, PATH_Y, PATH_Z]

    # Run the IMPROVED Max Jitter simulation
    run_stasis_generation_max_jitter(QUERY_EXAMPLE, simulated_paths, MU_JITTER_PENALTY)
