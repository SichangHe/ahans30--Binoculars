from typing import Final

BINOCULARS_ACCURACY_THRESHOLD: Final = 0.9015310749276843
"""Score threshold optimized for f1-score, selected using Falcon-7B and Falcon-7B-Instruct at bfloat16"""
BINOCULARS_FPR_THRESHOLD: Final = 0.8536432310785527
"""Score threshold optimized for low-fpr (chosen at 0.01%), selected using Falcon-7B and Falcon-7B-Instruct at bfloat16"""
CONTEXT_WINDOW: Final = 2048
"""Falcon-7B context window size."""
