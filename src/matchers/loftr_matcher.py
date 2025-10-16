"""
LoFTR Matcher (Semi-Dense Learned Matcher)
"""

import numpy as np
import torch
from typing import Dict, Optional
from .base_matcher import BaseMatcher


class LoFTRMatcher(BaseMatcher):
    """
    LoFTR - Detector-Free Local Feature Matching with Transformers
    Semi-dense learned matcher, không cần extractor riêng
    """
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        super().__init__(config, device)
        
        matcher_config = config.get('matcher', {}).get('loftr', {})
        
        # Load LoFTR
        try:
            from kornia.feature import LoFTR as KorniaLoFTR
            
            model_type = matcher_config.get('model_type', 'outdoor')
            self.match_threshold = matcher_config.get('match_threshold', 0.2)
            
            pretrained_type = 'outdoor' if model_type == 'outdoor' else 'indoor'
            
            self.loftr = KorniaLoFTR(pretrained=pretrained_type).to(self.device).eval()
            self.available = True
            
        except ImportError:
            print("Warning: kornia not available, LoFTR matcher disabled")
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
        Match using LoFTR (end-to-end, không cần keypoints/descriptors đầu vào)
        """
        if not self.available:
            return {
                'keypoints0': np.zeros((0, 2), dtype=np.float32),
                'keypoints1': np.zeros((0, 2), dtype=np.float32),
                'confidence': np.zeros((0,), dtype=np.float32)
            }
        
        # Convert images to grayscale tensors
        if len(image0.shape) == 3:
            gray0 = image0.mean(axis=2)
            gray1 = image1.mean(axis=2)
        else:
            gray0 = image0
            gray1 = image1
        
        # Normalize to [0, 1]
        gray0 = gray0.astype(np.float32) / 255.0
        gray1 = gray1.astype(np.float32) / 255.0
        
        # To tensors (1, 1, H, W)
        tensor0 = torch.from_numpy(gray0).unsqueeze(0).unsqueeze(0).to(self.device)
        tensor1 = torch.from_numpy(gray1).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Match
        with torch.no_grad():
            input_dict = {
                'image0': tensor0,
                'image1': tensor1
            }
            correspondences = self.loftr(input_dict)
        
        # Extract matches
        mkpts0 = correspondences['keypoints0'].cpu().numpy()  # (N, 2)
        mkpts1 = correspondences['keypoints1'].cpu().numpy()  # (N, 2)
        confidence = correspondences['confidence'].cpu().numpy()  # (N,)
        
        # Filter by confidence threshold
        mask = confidence > self.match_threshold
        mkpts0 = mkpts0[mask]
        mkpts1 = mkpts1[mask]
        confidence = confidence[mask]
        
        return {
            'keypoints0': mkpts0.astype(np.float32),
            'keypoints1': mkpts1.astype(np.float32),
            'confidence': confidence.astype(np.float32)
        }
    
    @property
    def name(self) -> str:
        return "LoFTR"
    
    @property
    def is_learned(self) -> bool:
        return True

