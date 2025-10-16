"""
SIFT Feature Extractor (Classical)
"""

import cv2
import numpy as np
from typing import Dict
from .base_extractor import BaseExtractor


class SIFTExtractor(BaseExtractor):
    """
    SIFT (Scale-Invariant Feature Transform) extractor
    Hand-crafted features, robust to scale and rotation
    """
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        super().__init__(config, device)
        
        extractor_config = config.get('extractor', {}).get('sift', {})
        
        # SIFT parameters
        nfeatures = extractor_config.get('nfeatures', 0)  # 0 = unlimited
        nOctaveLayers = extractor_config.get('nOctaveLayers', 3)
        contrastThreshold = extractor_config.get('contrastThreshold', 0.04)
        edgeThreshold = extractor_config.get('edgeThreshold', 10)
        sigma = extractor_config.get('sigma', 1.6)
        
        self.sift = cv2.SIFT_create(
            nfeatures=nfeatures,
            nOctaveLayers=nOctaveLayers,
            contrastThreshold=contrastThreshold,
            edgeThreshold=edgeThreshold,
            sigma=sigma
        )
    
    def extract(self, image: np.ndarray) -> Dict:
        """
        Extract SIFT features
        
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
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        if descriptors is None or len(keypoints) == 0:
            return {
                'keypoints': np.zeros((0, 2), dtype=np.float32),
                'descriptors': np.zeros((0, 128), dtype=np.float32),
                'scores': np.zeros((0,), dtype=np.float32)
            }
        
        # Convert keypoints to array
        kpts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        scores = np.array([kp.response for kp in keypoints], dtype=np.float32)
        
        return {
            'keypoints': kpts,
            'descriptors': descriptors.astype(np.float32),
            'scores': scores
        }
    
    @property
    def name(self) -> str:
        return "SIFT"

