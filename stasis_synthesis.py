#!/usr/bin/env python3
import pandas as pd
import numpy as np
from probv3_1 import ProbabilityStasisV3_1

# --- CONFIGURATION ---
STASIS_THRESHOLD = 0.80  # Only use messages with a high stability score (V3.1 Score > 0.80)
ALPHA_TV_PENALTY = 0.5
BETA_TRANS_PENALTY = 0.1

# --- PATH GENERATION SIMULATION (Same as previous step) ---
def generate_simulated_path(message_body: str) -> np.ndarray:
    """
    Simulates a probability path based on message length and content complexity.
    """
    length = len(message_body)
    words = len(message_body.split())
    
    p0_base = np.clip(1.0 - (words / 30.0), 0.5, 0.95)
    complexity = sum(1 for char in message_body if not char.isalnum() and char not in (' ', ','))
    volatility_factor = np.clip(complexity / 10.0, 0.0, 0.5)

    p = [p0_base] * 4
    
    if length > 50:
        p[1] = np.clip(p0_base - 0.6 * volatility_factor - 0.1, 0.30, p0_base)
        p[2] = np.clip(p[1] + 0.5 * volatility_factor + 0.1, 0.60, 0.90)
    elif length > 20 and volatility_factor > 0.05:
        jitter = 0.05 + 0.1 * volatility_factor
        p[1] = np.clip(p0_base - jitter, 0.50, 0.95)
        p[3] = np.clip(p0_base - jitter, 0.50, 0.95)
        
    return np.array(p)

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    
    # Initialize the V3.1 Dynamic Filter
    stasis_engine = ProbabilityStasisV3_1(alpha_tv=ALPHA_TV_PENALTY, beta_trans=BETA_TRANS_PENALTY)
    
    try:
        df = pd.read_csv('chat_messages_copy.csv')
    except FileNotFoundError:
        print("Error: chat_messages_copy.csv not found.")
        exit()
        
    stable_messages = []
    
    print(f"--- 🧠 Starting V3.1 Stasis Synthesis ---")
    print(f"Filtering messages with V3.1 Score > {STASIS_THRESHOLD:.2f} for final output.")

    # 1. Score all messages and filter
    for index, row in df.iterrows():
        message = row['body_full']
        
        # Skip empty messages or very short noise that won't contribute (like single emojis)
        if len(message.strip()) < 3:
            continue
        
        probs = generate_simulated_path(message)
        score = stasis_engine.stasis_score(probs)
        
        if score > STASIS_THRESHOLD:
            stable_messages.append(message.strip())

    if not stable_messages:
        print("\n--- Synthesis Failed ---")
        print("No messages met the high stability threshold of V3.1 Score > 0.80.")
    else:
        # 2. Synthesize the text
        synthesized_text = " ".join(stable_messages)
        
        # 3. Final Output
        print("\n==================================================================")
        print(">>> FINAL V3.1 STASIS-SYNTHESIZED TEXT (High-Confidence Composite) <<<")
        print(f"Total stable messages used: {len(stable_messages)}")
        print("==================================================================")
        print(synthesized_text)
        print("\n==================================================================")

