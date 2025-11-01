#!/usr/bin/env python3
"""
Final V2 generation demonstration using the improved probv2.py (Total Variation).
"""

from probv2 import ProbabilityStasisV2
from typing import List, Tuple
from tabulate import tabulate
import numpy as np

# --- HYPERPARAMETER ---
ALPHA_TV_PENALTY = 0.5 

# Define the structure of a reasoning path
ReasoningPath = Tuple[str, List[float], List[str]]

def run_final_generation_v2(query: str, paths: List[ReasoningPath], alpha_val: float, max_keep: int = 1):
    """Simulates LLM path generation, filters paths using V2 Stasis, and selects the final answer."""
    print(f"--- 💡 Query: {query} ---")

    # 1. Instantiate the V2 Stasis Engine
    stasis_engine = ProbabilityStasisV2(alpha_tv_penalty=alpha_val, max_keep=max_keep)

    # 2. Extract paths for filtering and scoring
    paths_for_filtering = [(name, probs) for name, probs, _ in paths]
    scored_paths = stasis_engine.filter_paths(paths_for_filtering)

    # 3. Compile results for display
    analysis_results = []
    
    for rank, (name, score) in enumerate(scored_paths):
        original_path = next((p for p in paths if p[0] == name), None)
        probs = np.array(original_path[1])
        
        # Calculate components for display
        mean_val = np.mean(probs)
        total_variation = np.sum(np.abs(probs[1:] - probs[:-1])) if len(probs) > 1 else 0.0
        penalty = alpha_val * total_variation

        analysis_results.append([
            rank + 1,
            name,
            f"{mean_val:.4f}",
            f"{total_variation:.4f}",
            f"{penalty:.4f}",
            f"{score:.4f}"
        ])

    # Display Analysis
    headers = ["Rank", "Path", "Mean", "Total Variation", f"Penalty (α={alpha_val}*TV)", "Stasis Score"]
    print(f"--- 🌟 Total Variation Filter (V2) ---")
    print(tabulate(analysis_results, headers=headers, tablefmt="fancy_grid"))

    # 4. Select and Generate Final Text
    winning_path_name = scored_paths[0][0]
    winning_path_data = next((p for p in paths if p[0] == winning_path_name), None)
    final_answer = winning_path_data[2][-1]

    print(f"\n[Step 2: Final Answer Selection]")
    print(f"🏆 Winning Path (Highest V2 Stasis Score): {winning_path_name}")
    print("\n=========================================================")
    print(f"FINAL V2 STASIS-APPROVED GENERATION:")
    print(f"{final_answer}")
    print("=========================================================\n")

if __name__ == '__main__':
    QUERY = "Explain the two main theories for how the Moon was formed."
    
    # Path A: Fission Theory (Stable, low confidence) - TV = 0.10
    PATH_A = (
        "Path A: Fission Theory (Stable/Low Mean)",
        [0.60, 0.65, 0.60],
        ["...","Final Answer: The Fission Theory suggests the Moon spun out of Earth, though this is less favored today."]
    )

    # Path B: Giant Impact Theory (High Mean, Jumpy) - TV = 0.90
    PATH_B = (
        "Path B: Giant Impact (Jumpy/Unreliable)",
        [0.95, 0.50, 0.90, 0.85],
        ["...","Final Answer: The Giant Impact Theory (Theia), but LLM showed severe confusion."]
    )

    # Path C: Giant Impact Theory (Stable, High Confidence) - TV = 0.10
    PATH_C = (
        "Path C: Giant Impact (Stable/Reliable)",
        [0.92, 0.95, 0.90, 0.92],
        ["...","Final Answer: The Giant Impact Theory is the dominant explanation, positing that a Mars-sized object's collision with early Earth created the Moon from ejected debris."]
    )
    
    simulated_paths = [PATH_A, PATH_B, PATH_C]

    # Run the final V2 generation simulation
    run_final_generation_v2(QUERY, simulated_paths, ALPHA_TV_PENALTY, max_keep=3)
