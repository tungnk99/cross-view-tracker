"""
Pipeline Factory - Khởi tạo pipeline từ config
"""

from typing import Dict
from .base_pipeline import BasePipeline
from .classical_pipeline import ClassicalPipeline
from .semidense_pipeline import SemiDensePipeline
from .e2e_pipeline import E2EPipeline
from ..utils import load_config


def create_pipeline(
    config_path: str = None,
    config: Dict = None,
    pipeline_type: str = None,
    device: str = 'cuda'
) -> BasePipeline:
    """
    Factory function to create pipeline
    
    Args:
        config_path: Path to YAML config file
        config: Config dictionary (if not loading from file)
        pipeline_type: Pipeline type ('classical', 'semi_dense', 'e2e')
                      If None, will read from config
        device: Device to run on ('cuda' or 'cpu')
        
    Returns:
        Initialized pipeline instance
        
    Examples:
        # Load from config file
        pipeline = create_pipeline('configs/classical.yaml')
        
        # Specify pipeline type explicitly
        pipeline = create_pipeline('configs/default.yaml', pipeline_type='e2e')
        
        # Use config dict
        config = {...}
        pipeline = create_pipeline(config=config, pipeline_type='classical')
    """
    # Load config
    if config is None:
        if config_path is None:
            raise ValueError("Either config_path or config must be provided")
        config = load_config(config_path)
    
    # Determine pipeline type
    if pipeline_type is None:
        # Try to read from config
        pipeline_type = config.get('pipeline', {}).get('type', 'classical')
    
    # Create pipeline
    if pipeline_type == 'classical':
        print(f"\n{'='*60}")
        print("Creating Classical Pipeline (Sparse Feature-Based)")
        print("Extractor → Matcher → Mapper")
        print(f"{'='*60}")
        return ClassicalPipeline(config, device)
    
    elif pipeline_type == 'semi_dense':
        print(f"\n{'='*60}")
        print("Creating Semi-Dense Pipeline (Semi-Dense Matcher)")
        print("Semi-Dense Matcher → Mapper")
        print("LoFTR/Efficient LoFTR produces semi-dense correspondences")
        print(f"{'='*60}")
        return SemiDensePipeline(config, device)
    
    elif pipeline_type == 'e2e':
        print(f"\n{'='*60}")
        print("Creating E2E Pipeline (Dense Correspondence)")
        print("Dense Model → Direct Mapping (no explicit RANSAC)")
        print("For true E2E: DKM, PDC-Net, GLU-Net")
        print(f"{'='*60}")
        return E2EPipeline(config, device)
    
    else:
        raise ValueError(
            f"Unknown pipeline type: {pipeline_type}. "
            f"Must be one of: 'classical', 'semi_dense', 'e2e'"
        )


def list_available_pipelines() -> Dict:
    """
    List all available pipeline types and their components
    
    Returns:
        Dictionary with pipeline information
    """
    return {
        'classical': {
            'name': 'Classical Pipeline (Sparse Feature-Based)',
            'description': 'Extractor → Matcher → Mapper',
            'extractors': ['sift', 'orb', 'superpoint'],
            'matchers': ['brute_force', 'flann'],
            'mappers': ['homography', 'affine'],
            'characteristics': [
                'Sparse features',
                'Multiple steps',
                'Interpretable',
                'No training required'
            ]
        },
        'semi_dense': {
            'name': 'Semi-Dense Pipeline (Semi-Dense Matcher)',
            'description': 'Semi-Dense Matcher → Mapper',
            'matchers': ['loftr (semi-dense)', 'efficient_loftr (semi-dense)', 'lightglue', 'superglue'],
            'mappers': ['homography', 'affine', 'learned'],
            'characteristics': [
                'Semi-dense correspondences (1000s of matches)',
                '2 steps (matcher + geometric fitting)',
                'Detector-free DL matching',
                'High accuracy with interpretability',
                'Recommended for most tasks'
            ]
        },
        'e2e': {
            'name': 'End-to-End Pipeline (Dense Correspondence)',
            'description': 'Dense Model → Direct Mapping',
            'models': ['dkm', 'pdcnet', 'glunet', 'raft'],
            'characteristics': [
                '1 step (no explicit RANSAC)',
                'Dense pixel-wise correspondence',
                'End-to-end learning',
                'Handles large cross-view gap',
                'Maximum robustness'
            ]
        }
    }

