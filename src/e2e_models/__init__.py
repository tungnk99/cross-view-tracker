"""
End-to-End Models - Dense/ROI-aware mapping
Hỗ trợ: DKM, MatchFormer, GMFlow, RAFT

Note: EfficientLoFTR has been moved to matchers since it's primarily a matcher.
For E2E pipelines, use EfficientLoFTRMatcher from matchers.
"""

from .base_e2e_model import BaseE2EModel

__all__ = [
    'BaseE2EModel'
]

