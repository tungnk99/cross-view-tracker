"""
SuperGlue Matcher (Semi-Dense Learned Matcher)
"""

import numpy as np
from typing import Dict, Optional
from .base_matcher import BaseMatcher


class SuperGlueMatcher(BaseMatcher):
    """
    SuperGlue - Learning Feature Matching with Graph Neural Networks
    Kết hợp với SuperPoint extractor
    
    Note: Requires external SuperGlue implementation
    """
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        super().__init__(config, device)
        
        matcher_config = config.get('matcher', {}).get('superglue', {})
        
        self.match_threshold = matcher_config.get('match_threshold', 0.2)
        
        # SuperGlue requires external implementation
        print("Warning: SuperGlue requires external implementation")
        print("Please install from: https://github.com/magicleap/SuperGluePretrainedNetwork")
        self.available = False
    
    def match(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        keypoints0: Optional[np.ndarray] = None,
        keypoints1: Optional[np.ndarray] = None,
        descriptors0: Optional[np.ndarray] = None,
        descriptors1: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Match using SuperPoint + SuperGlue
        
        Note: Placeholder implementation
        """
        if not self.available:
            return {
                'keypoints0': np.zeros((0, 2), dtype=np.float32),
                'keypoints1': np.zeros((0, 2), dtype=np.float32),
                'confidence': np.zeros((0,), dtype=np.float32)
            }
        
        # TODO: Implement SuperGlue matching
        # Requires external SuperGlue model
        
        raise NotImplementedError("SuperGlue requires external implementation")
    
    @property
    def name(self) -> str:
        return "SuperGlue"
    
    @property
    def is_learned(self) -> bool:
        return True

