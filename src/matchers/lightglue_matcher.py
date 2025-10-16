"""
LightGlue Matcher (Semi-Dense Learned Matcher)
"""

import numpy as np
import torch
from typing import Dict, Optional
from .base_matcher import BaseMatcher


class LightGlueMatcher(BaseMatcher):
    """
    LightGlue - Local Feature Matching at Light Speed
    Kết hợp với SuperPoint extractor
    """
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        super().__init__(config, device)
        
        matcher_config = config.get('matcher', {}).get('lightglue', {})
        
        # Load LightGlue
        try:
            from kornia.feature import LightGlue as KorniaLightGlue
            from kornia.feature import SuperPoint as KorniaSuperPoint
            
            self.match_threshold = matcher_config.get('match_threshold', 0.2)
            
            # SuperPoint extractor
            self.superpoint = KorniaSuperPoint(pretrained=True).to(self.device).eval()
            
            # LightGlue matcher
            self.lightglue = KorniaLightGlue(
                pretrained='superpoint',
                features='superpoint'
            ).to(self.device).eval()
            
            self.available = True
            
        except ImportError:
            print("Warning: kornia not available, LightGlue matcher disabled")
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
        Match using SuperPoint + LightGlue
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
        
        with torch.no_grad():
            # Extract features
            feats0 = self.superpoint({'image': tensor0})
            feats1 = self.superpoint({'image': tensor1})
            
            # Match
            matches = self.lightglue({
                'image0': feats0,
                'image1': feats1
            })
        
        # Extract matched keypoints
        mkpts0 = matches['keypoints0'].cpu().numpy()  # (N, 2)
        mkpts1 = matches['keypoints1'].cpu().numpy()  # (N, 2)
        
        # Confidence scores
        if 'scores' in matches:
            confidence = matches['scores'].cpu().numpy()
        else:
            confidence = np.ones(len(mkpts0), dtype=np.float32)
        
        # Filter by threshold
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
        return "LightGlue"
    
    @property
    def is_learned(self) -> bool:
        return True

