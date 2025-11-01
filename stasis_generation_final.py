#!/usr/bin/env python3
"""
Final text generation demonstration using the improved ProbabilityStasis filter.
"""

from stasis_core import ProbabilityStasis
from typing import List, Tuple
from tabulate import tabulate
import numpy as np

# Define the structure of a reasoning path
ReasoningPath = Tuple[str, List[float], List[str]]

def run_final_generation(query: str, paths: List[ReasoningPath], mu_val: float = 1.5, max_keep: int = 1):
    """Simulates LLM path generation, filters paths using Stasis, and selects the final answer."""
    print(f"--- 💡 Query: {query} ---")

    # 1. Instantiate the Stasis Engine
    stasis_engine = ProbabilityStasis(mu_jitter_penalty=mu_val, max_keep=max_keep)

    # 2. Extract paths for filtering
    paths_for_filtering = [(name, probs) for name, probs, _ in paths]
    scored_paths = stasis_engine.filter_paths(paths_for_filtering)

    # 3. Compile results for display
    analysis_results = []
    
    for rank, (name, score) in enumerate(scored_paths):
        original_path = next((p for p in paths if p[0] == name), None)
        probs = np.array(original_path[1])
        
        # Calculate components for display (using the logic from the class for consistency)
        mean_val = np.mean(probs)
        drops = probs[:-1] - probs[1:]
        max_drop = np.max(np.concatenate(([0.0], drops[drops > 0])))
        penalty = mu_val * max_drop

        analysis_results.append([
            rank + 1,
            name,
            f"{mean_val:.4f}",
            f"{max_drop:.4f}",
            f"{penalty:.4f}",
            f"{score:.4f}"
        ])

    # Display Analysis
    headers = ["Rank", "Path", "Mean", "Max Drop", f"Penalty (μ={mu_val}*Drop)", "Stasis Score"]
    print(f"--- 🌟 Max Jitter Filter (μ={mu_val}) ---")
    print(tabulate(analysis_results, headers=headers, tablefmt="fancy_grid"))

    # 4. Select and Generate Final Text
    winning_path_name = scored_paths[0][0]
    winning_path_data = next((p for p in paths if p[0] == winning_path_name), None)
    final_answer = winning_path_data[2][-1]

    print(f"\n[Step 2: Final Answer Selection]")
    print(f"🏆 Winning Path (Highest Stasis Score): {winning_path_name}")
    print("\n=========================================================")
    print(f"FINAL STASIS-APPROVED GENERATION:")
    print(f"{final_answer}")
    print("=========================================================\n")

if __name__ == '__main__':
    QUERY = "Explain the two main theories for how the Moon was formed."
    
    # Path A: Fission Theory (Stable, low confidence)
    PATH_A = (
        "Path A: Fission Theory (Low Confidence/Stable)",
        [0.60, 0.65, 0.60],
        ["The Fission theory proposes the Moon broke away from a rapidly spinning Earth.", "It is currently considered less likely due to angular momentum issues.", "Final Answer: The Fission Theory suggests the Moon spun out of Earth, though this is less favored today."]
    )

    # Path B: Giant Impact Theory (High Mean, Jumpy)
    PATH_B = (
        "Path B: Giant Impact (Jumpy/Unreliable)",
        [0.95, 0.50, 0.90, 0.85], # Massive drop from 0.95 to 0.50
        ["The Giant Impact theory is the leading model.", "Wait, some older models are still mentioned (confidence drop).", "It involves a Mars-sized object named Theia hitting early Earth.", "Final Answer: The Giant Impact Theory (Theia) is the widely accepted model for lunar formation, but LLM showed internal confusion."]
    )

    # Path C: Giant Impact Theory (Stable, High Confidence)
    PATH_C = (
        "Path C: Giant Impact (Stable/Reliable)",
        [0.92, 0.95, 0.90, 0.92],
        ["The Giant Impact theory suggests a protoplanet named Theia struck Earth.", "This collision ejected material that coalesced into the Moon.", "This theory is supported by the isotopic similarity of Moon and Earth rocks.", "Final Answer: The Giant Impact Theory is the dominant explanation, positing that a Mars-sized object's collision with early Earth created the Moon from ejected debris."]
    )
    
    simulated_paths = [PATH_A, PATH_B, PATH_C]

    # Run the final generation simulation
    run_final_generation(QUERY, simulated_paths, mu_val=1.5, max_keep=3)
