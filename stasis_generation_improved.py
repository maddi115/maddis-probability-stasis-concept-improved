#!/usr/bin/env python3
"""
Demonstrates the IMPROVED Probability Stasis filtering with Skewness Bonus.
"""

import numpy as np
from tabulate import tabulate
from typing import List, Tuple, Dict

# --- NEW HYPERPARAMETERS ---
LAMBDA_INSTABILITY = 1.0 # Penalty for Variance + Range
BETA_SKEWNESS = 0.25     # Bonus/Penalty for Skewness

# Define the structure of a reasoning path
ReasoningPath = Tuple[str, List[float], List[str]]

def improved_stasis_score(probs: List[float], lambda_val: float, beta_val: float) -> float:
    """Calculates the IMPROVED Stasis Score with Skewness Bonus/Penalty."""
    p = np.array(probs)
    
    # 1. Mean (Confidence)
    mean_val = np.mean(p)
    
    # 2. Instability Penalty Components
    var_val = np.var(p)
    range_val = np.max(p) - np.min(p)
    instability_penalty = lambda_val * (var_val + range_val)
    
    # 3. Skewness Component (The Improvement)
    # The moment-based skewness function from scipy/numpy is best here. 
    # For a small sequence, this captures if the probabilities trended up or down.
    skewness_val = (np.mean((p - mean_val)**3) / np.std(p)**3) if np.std(p) != 0 else 0
    
    # NEW FORMULA: Score = Mean - Penalty + Skewness_Bonus
    score = mean_val - instability_penalty + beta_val * skewness_val
    return score, mean_val, var_val, range_val, instability_penalty, skewness_val

def run_stasis_generation_improved(query: str, paths: List[ReasoningPath], lambda_val: float, beta_val: float):
    """Runs the improved simulation."""
    print(f"--- 💡 Query: {query} ---")
    print(f"--- 🌟 IMPROVED Stasis Filter (λ={lambda_val}, β={beta_val}) ---")

    # 1. Score all paths
    analysis_results = []
    
    for name, probs, text_steps in paths:
        score, mean_val, var_val, range_val, penalty, skew_val = improved_stasis_score(probs, lambda_val, beta_val)
        
        analysis_results.append({
            "name": name,
            "score": score,
            "mean": f"{mean_val:.4f}",
            "variance": f"{var_val:.4f}",
            "range": f"{range_val:.4f}",
            "skewness": f"{skew_val:.4f}",
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
            data['variance'],
            data['range'],
            data['skewness'],
            data['penalty'],
            f"{data['score']:.4f}"
        ])
    
    headers = ["Rank", "Path", "Mean", "Var", "Range", "Skewness (Bonus)", "Penalty (λ*Instability)", "New StasisScore"]
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

    # 3. Select and Generate Final Text
    winning_path_name = sorted_results[0]['name']
    winning_steps = sorted_results[0]['text_steps']
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
        ["...Correct answer steps...","Final Answer: Canberra, stable reasoning."]
    )

    # Path Y: High Fluctuation (Jumpy Correct) - Expected HUGE Skewness Bonus
    PATH_Y = (
        "Path Y: High Fluctuation (Jumpy Correct)",
        [0.55, 0.99, 0.40, 0.88], # Mean is 0.705. Starts low, ends high (positive skew).
        ["...Correct answer steps...","Final Answer: Canberra, jumpy reasoning that ended high."]
    )

    # Path Z: Low Stasis (Stable Wrong)
    PATH_Z = (
        "Path Z: Stable Wrong (Incorrect)",
        [0.70, 0.75, 0.70, 0.75],
        ["...Incorrect answer steps...","Final Answer: Sydney, stable reasoning."]
    )
    
    simulated_paths = [PATH_X, PATH_Y, PATH_Z]

    # Run the IMPROVED simulation
    run_stasis_generation_improved(QUERY_EXAMPLE, simulated_paths, LAMBDA_INSTABILITY, BETA_SKEWNESS)
