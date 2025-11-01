#!/usr/bin/env python3
"""
Test suite for ProbabilityStasis engine.
"""

import unittest
from stasis_core import ProbabilityStasis
import numpy as np

class TestProbabilityStasis(unittest.TestCase):
    
    def setUp(self):
        # Default engine with lambda = 1.0
        self.ps_default = ProbabilityStasis() 
        # Engine with high penalty (lambda = 5.0)
        self.ps_high_penalty = ProbabilityStasis(lambda_instability=5.0) 

    def test_stasis_score_perfect_stasis(self):
        # Probs: [0.9, 0.9, 0.9]. Mean=0.9, Var=0, Range=0. Score = 0.9 - 1*(0+0) = 0.9
        self.assertAlmostEqual(self.ps_default.stasis_score([0.9, 0.9, 0.9]), 0.9)

    def test_stasis_score_high_variance(self):
        # Probs: [0.1, 0.9, 0.1, 0.9]. Mean=0.5, Var=0.16, Range=0.8. Penalty = 0.96
        # Score = 0.5 - 1*(0.16 + 0.8) = -0.46  <-- CORRECTED VALUE
        score = self.ps_default.stasis_score([0.1, 0.9, 0.1, 0.9])
        self.assertAlmostEqual(score, -0.46) # <-- FIXED ASSERTION

    def test_stasis_score_high_penalty_effect(self):
        # Probs: [0.7, 0.75, 0.8]. Mean=0.75, Var=0.00166..., Range=0.1. Penalty = 0.10166...
        # Default: Score = 0.75 - 1*(0.10166...) = 0.6483
        # High Penalty: Score = 0.75 - 5*(0.10166...) = 0.2417
        probs = [0.7, 0.75, 0.8]
        default_score = self.ps_default.stasis_score(probs)
        high_penalty_score = self.ps_high_penalty.stasis_score(probs)
        
        self.assertAlmostEqual(default_score, 0.6483333333333333)
        self.assertAlmostEqual(high_penalty_score, 0.2416666666666666)
        self.assertTrue(high_penalty_score < default_score)

    def test_filter_paths(self):
        paths = [
            ("A_Perfect", [0.9, 0.9, 0.9]), # Score 0.9
            ("B_Fickle", [0.9, 0.1]),       # Score -0.46 (Mean 0.5, Var 0.16, Range 0.8)
            ("C_Stable", [0.7, 0.7, 0.7]),  # Score 0.7
        ]
        
        # Test default max_keep=3
        filtered_3 = self.ps_default.filter_paths(paths)
        self.assertEqual(len(filtered_3), 3)
        self.assertEqual(filtered_3[0][0], "A_Perfect")
        
        # Test max_keep=1
        ps_keep_1 = ProbabilityStasis(max_keep=1)
        filtered_1 = ps_keep_1.filter_paths(paths)
        self.assertEqual(len(filtered_1), 1)
        self.assertEqual(filtered_1[0][0], "A_Perfect")

    def test_single_probability(self):
        # Variance/Range cannot be computed; returns 0.0 as per implementation
        self.assertEqual(self.ps_default.stasis_score([0.9]), 0.0)

if __name__ == '__main__':
    unittest.main(exit=False)
