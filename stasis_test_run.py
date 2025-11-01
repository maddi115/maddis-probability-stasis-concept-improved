#!/usr/bin/env python3
import numpy as np
from stasis_core import ProbabilityStasis
from tabulate import tabulate
from typing import List, Tuple

# Instantiate the engine with lambda=2.0 (high penalty for instability) and max_keep=3
stasis_engine = ProbabilityStasis(lambda_instability=2.0, max_keep=3)

# The Mock LLM Reasoning Dataset
full_dataset = [
    ("P1: Perfect Stasis", [0.9, 0.9, 0.9, 0.9]),
    ("P2: High Mean, Low Range", [0.8, 0.85, 0.78, 0.82]),
    ("P3: High Fluctuation", [0.9, 0.1, 0.9, 0.1]),
    ("P4: Steady, Low Mean", [0.55, 0.5, 0.52, 0.53]),
    ("P5: Declining Confidence", [0.9, 0.8, 0.7, 0.6]),
    ("P6: Ascending Confidence", [0.6, 0.7, 0.8, 0.9]),
    ("P7: Moderate Jitter", [0.7, 0.8, 0.7, 0.8]),
    ("P8: Single Step", [0.95]), # Should result in score 0.0 (too short)
]

print("--- 🧠 Probability Stasis Analysis: Full Dataset ---")

results = []
for name, probs in full_dataset:
    p = np.array(probs)
    score = stasis_engine.stasis_score(probs)
    
    # Calculate components for display
    mean_val = np.mean(p) if len(p) >= 2 else 0.0
    var_val = np.var(p) if len(p) >= 2 else 0.0
    range_val = np.max(p) - np.min(p) if len(p) >= 2 else 0.0
    instability_penalty = stasis_engine.lambda_instability * (var_val + range_val)
    
    results.append([
        name,
        ", ".join(f"{x:.2f}" for x in probs),
        f"{mean_val:.4f}",
        f"{var_val:.4f}",
        f"{range_val:.4f}",
        f"2.0 * ({var_val:.4f} + {range_val:.4f})",
        f"{instability_penalty:.4f}",
        f"{score:.4f}"
    ])

# Print the results table
headers = ["Path", "Probs", "Mean", "Variance", "Range", "Penalty Calc", "Penalty", "StasisScore"]
print(tabulate(results, headers=headers, tablefmt="fancy_grid"))

# Filter and display the survivors
survivors = stasis_engine.filter_paths(full_dataset)

print("\n--- 🏆 The Eternal Survivors (Top 3) ---")
for i, (name, score) in enumerate(survivors):
    print(f"  {i+1}. {name} (Score: {score:.4f})")
print("---------------------------------------")

