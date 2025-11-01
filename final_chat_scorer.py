#!/usr/bin/env python3
import pandas as pd
import numpy as np
from tabulate import tabulate
from probv3_1 import ProbabilityStasisV3_1

# --- CONFIGURATION ---
# Base hyperparameters for V3.1 (same as our test values)
ALPHA_TV_PENALTY = 0.5
BETA_TRANS_PENALTY = 0.1

# --- PATH GENERATION SIMULATION FUNCTION ---
def generate_simulated_path(message_body: str) -> np.ndarray:
    """
    Simulates a probability path based on message length and content complexity.
    This function must match the logic used in our initial validation.
    """
    length = len(message_body)
    words = len(message_body.split())
    
    # 1. Determine Mean/P0 based on length/simplicity
    # Shorter, simpler messages get a higher P0
    p0_base = np.clip(1.0 - (words / 30.0), 0.5, 0.95)
    
    # 2. Determine Volatility based on complexity (e.g., presence of symbols/punctuation)
    complexity = sum(1 for char in message_body if not char.isalnum() and char not in (' ', ','))
    volatility_factor = np.clip(complexity / 10.0, 0.0, 0.5)

    # 3. Generate a path: Start high, then introduce volatility
    # This simulation favors stability for short, direct messages.
    p = [p0_base] * 4
    
    if length > 50:
        # Long, complex messages (like Path A) get a severe drop
        p[1] = np.clip(p0_base - 0.6 * volatility_factor - 0.1, 0.30, p0_base)
        p[2] = np.clip(p[1] + 0.5 * volatility_factor + 0.1, 0.60, 0.90)
    elif length > 20 and volatility_factor > 0.05:
        # Medium, jittery messages (like Path B/C) get small jitters
        jitter = 0.05 + 0.1 * volatility_factor
        p[1] = np.clip(p0_base - jitter, 0.50, 0.95)
        p[3] = np.clip(p0_base - jitter, 0.50, 0.95)
        
    return np.array(p)

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    
    # Initialize the V3.1 Dynamic Filter
    stasis_engine = ProbabilityStasisV3_1(alpha_tv=ALPHA_TV_PENALTY, beta_trans=BETA_TRANS_PENALTY)
    
    # Load the chat data
    try:
        df = pd.read_csv('chat_messages_copy.csv')
    except FileNotFoundError:
        print("Error: chat_messages_copy.csv not found.")
        exit()
        
    results = []
    
    # Iterate through all messages
    for index, row in df.iterrows():
        message = row['body_full']
        
        # 1. Generate simulated path based on message content
        probs = generate_simulated_path(message)
        
        # 2. Score the path using the V3.1 Dynamic Filter
        score = stasis_engine.stasis_score(probs)
        
        # 3. Calculate metrics for display
        p0 = probs[0]
        mean = np.mean(probs)
        
        # Calculate dynamic penalties for full transparency
        alpha_dyn, beta_dyn = stasis_engine._get_dynamic_weights(p0)
        tv_penalty = alpha_dyn * np.sum(np.abs(probs[1:] - probs[:-1])) if len(probs) > 1 else 0.0
        transition_penalty = beta_dyn * stasis_engine._calculate_transition_penalty(probs)
        
        
        results.append({
            'Score': score,
            'P0': p0,
            'Mean': mean,
            'TV_Pen': tv_penalty,
            'Trans_Pen': transition_penalty,
            'Message': message[:60] + '...' if len(message) > 60 else message
        })

    # Sort and Display Results
    results.sort(key=lambda x: x['Score'], reverse=True)
    
    table_data = []
    for rank, res in enumerate(results[:13], 1):
        table_data.append([
            rank,
            f"{res['Score']:.4f}",
            f"{res['P0']:.2f}",
            f"{res['Mean']:.2f}",
            f"{res['TV_Pen']:.4f}",
            f"{res['Trans_Pen']:.4f}",
            res['Message']
        ])

    print("\n--- 🏆 V3.1 DYNAMIC FILTER STASIS RANKING on chat_messages_copy.csv ---")
    print("Scores reflect the inferred stability of LLM reasoning based on message complexity.")
    print(tabulate(table_data, headers=['Rank', 'V3.1 Score', 'P0', 'Mean', 'TV Pen', 'Trans Pen', 'Message Snippet'], tablefmt="fancy_grid"))

