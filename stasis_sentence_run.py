#!/usr/bin/env python3
import numpy as np
from stasis_core import ProbabilityStasis
from tabulate import tabulate
from typing import List, Tuple

# Instantiate the engine with lambda=1.0 (LOWER penalty for instability) and max_keep=3
stasis_engine = ProbabilityStasis(lambda_instability=1.0, max_keep=3)

# The Simulated Sentence Reasoning Paths (Data remains the same)
full_dataset = [
    ("P1: Sun rises in the east", [0.95, 0.95, 0.95, 0.95]),
    ("P2: Coffee & news", [0.85, 0.90, 0.85, 0.90]),
    ("P3: Dog barks loudly", [0.70, 0.90, 0.50, 0.80]),
    ("P4: Japan vacation", [0.60, 0.90, 0.10, 0.40]),
    ("P5: Apples are healthy", [0.90, 0.85, 0.85, 0.90]),
    ("P6: Forgot umbrella", [0.80, 0.60, 0.70, 0.50]),
    ("P7: Library closes at 8", [0.65, 0.70, 0.65, 0.70]),
    ("P8: Learning language", [0.70, 0.85, 0.75, 0.80]),
    ("P9: Deep-sea creatures", [0.10, 0.90, 0.20, 0.80]),
    ("P10: Recycling", [0.99, 0.98, 0.99, 0.98]),
]

print("--- 🧠 Probability Stasis Analysis: Sentence Dataset (Lambda = 1.0) ---")

results = []
for name, probs in full_dataset:
    p = np.array(probs)
    score = stasis_engine.stasis_score(probs)
    
    # Calculate components for display
    mean_val = np.mean(p)
    var_val = np.var(p)
    range_val = np.max(p) - np.min(p)
    instability_penalty = stasis_engine.lambda_instability * (var_val + range_val)
    
    results.append([
        name,
        f"{mean_val:.4f}",
        f"{var_val:.4f}",
        f"{range_val:.4f}",
        f"{instability_penalty:.4f}",
        f"{score:.4f}"
    ])

# Print the results table
headers = ["Path Name", "Mean", "Variance", "Range", "Penalty (1*Instability)", "StasisScore"]
print(tabulate(results, headers=headers, tablefmt="fancy_grid"))

# Filter and display the survivors
survivors = stasis_engine.filter_paths(full_dataset)

print("\n--- 🏆 The Eternal Survivors (Top 3 Consistent Paths) ---")
for i, (name, score) in enumerate(survivors):
    print(f"  {i+1}. {name} (Score: {score:.4f})")
print("---------------------------------------------------------")
