"""
ORB Feature Extractor (Classical)
"""

import cv2
import numpy as np
from typing import Dict
from .base_extractor import BaseExtractor


class ORBExtractor(BaseExtractor):
    """
    ORB (Oriented FAST and Rotated BRIEF) extractor
    Fast, binary descriptor, rotation invariant
    """
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        super().__init__(config, device)
        
        extractor_config = config.get('extractor', {}).get('orb', {})
        
        # ORB parameters
        nfeatures = extractor_config.get('nfeatures', 500)
        scaleFactor = extractor_config.get('scaleFactor', 1.2)
        nlevels = extractor_config.get('nlevels', 8)
        edgeThreshold = extractor_config.get('edgeThreshold', 31)
        
        self.orb = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=scaleFactor,
            nlevels=nlevels,
            edgeThreshold=edgeThreshold
        )
    
    def extract(self, image: np.ndarray) -> Dict:
        """
        Extract ORB features
        
        Args:
            image: RGB image (H, W, 3) or grayscale (H, W)
            
        Returns:
            Dict with keypoints, descriptors, scores
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect and compute
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if descriptors is None or len(keypoints) == 0:
            return {
                'keypoints': np.zeros((0, 2), dtype=np.float32),
                'descriptors': np.zeros((0, 32), dtype=np.uint8),
                'scores': np.zeros((0,), dtype=np.float32)
            }
        
        # Convert keypoints to array
        kpts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        scores = np.array([kp.response for kp in keypoints], dtype=np.float32)
        
        return {
            'keypoints': kpts,
            'descriptors': descriptors,  # uint8 binary descriptor
            'scores': scores
        }
    
    @property
    def name(self) -> str:
        return "ORB"

