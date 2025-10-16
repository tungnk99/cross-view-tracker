"""
Pipeline package - Hỗ trợ 3 loại pipeline:
1. Classical Pipeline (sparse feature-based)
2. Semi-Dense Pipeline (learned matcher) 
3. E2E Pipeline (dense/ROI-aware)
"""

from .base_pipeline import BasePipeline
from .classical_pipeline import ClassicalPipeline
from .semidense_pipeline import SemiDensePipeline
from .e2e_pipeline import E2EPipeline
from .factory import create_pipeline

__all__ = [
    'BasePipeline',
    'ClassicalPipeline',
    'SemiDensePipeline',
    'E2EPipeline',
    'create_pipeline'
]

