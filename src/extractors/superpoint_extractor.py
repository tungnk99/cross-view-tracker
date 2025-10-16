"""
SuperPoint Feature Extractor (Learned)
"""

import numpy as np
import torch
from typing import Dict
from .base_extractor import BaseExtractor


class SuperPointExtractor(BaseExtractor):
    """
    SuperPoint - Self-Supervised Interest Point Detection
    Learned feature extractor with high repeatability
    """
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        super().__init__(config, device)
        
        extractor_config = config.get('extractor', {}).get('superpoint', {})
        
        # Load SuperPoint from kornia
        try:
            from kornia.feature import SuperPoint as KorniaSuperPoint
            
            max_num_keypoints = extractor_config.get('max_keypoints', 1024)
            detection_threshold = extractor_config.get('detection_threshold', 0.005)
            
            self.superpoint = KorniaSuperPoint(
                pretrained=True,
                max_num_keypoints=max_num_keypoints,
                detection_threshold=detection_threshold
            ).to(self.device).eval()
            
            self.available = True
        except ImportError:
            print("Warning: kornia not available, SuperPoint extractor disabled")
            self.available = False
    
    def extract(self, image: np.ndarray) -> Dict:
        """
        Extract SuperPoint features
        
        Args:
            image: RGB image (H, W, 3) or grayscale (H, W)
            
        Returns:
            Dict with keypoints, descriptors, scores
        """
        if not self.available:
            return {
                'keypoints': np.zeros((0, 2), dtype=np.float32),
                'descriptors': np.zeros((0, 256), dtype=np.float32),
                'scores': np.zeros((0,), dtype=np.float32)
            }
        
        # Convert to grayscale tensor
        if len(image.shape) == 3:
            gray = image.mean(axis=2)
        else:
            gray = image
        
        # Normalize to [0, 1]
        gray = gray.astype(np.float32) / 255.0
        
        # To tensor (1, 1, H, W)
        tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.superpoint(tensor)
        
        # Extract results
        keypoints = features['keypoints'][0].cpu().numpy()  # (N, 2)
        descriptors = features['descriptors'][0].cpu().numpy().T  # (N, 256)
        scores = features['scores'][0].cpu().numpy()  # (N,)
        
        return {
            'keypoints': keypoints.astype(np.float32),
            'descriptors': descriptors.astype(np.float32),
            'scores': scores.astype(np.float32)
        }
    
    @property
    def name(self) -> str:
        return "SuperPoint"

