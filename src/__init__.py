"""
Cross-View Tracker System - 3-Pipeline Architecture

Supports 3 types of pipelines:
1. Classical Pipeline (Sparse Feature-Based)
2. Semi-Dense Pipeline (Learned Matcher)
3. End-to-End Pipeline (Dense/ROI-aware)
"""

__version__ = "2.0.0"

# Export pipeline factory
from .pipelines import create_pipeline, list_available_pipelines

# Export pipeline classes
from .pipelines import (
    BasePipeline,
    ClassicalPipeline,
    SemiDensePipeline,
    E2EPipeline
)

__all__ = [
    'create_pipeline',
    'list_available_pipelines',
    'BasePipeline',
    'ClassicalPipeline',
    'SemiDensePipeline',
    'E2EPipeline'
]

