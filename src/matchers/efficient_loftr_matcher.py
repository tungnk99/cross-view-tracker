"""
Efficient LoFTR Matcher - ROI-aware Dense Matching
Extends basic LoFTR with ROI-aware aggregation for direct point mapping
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
from .base_matcher import BaseMatcher


class EfficientLoFTRMatcher(BaseMatcher):
    """
    Efficient LoFTR - Dense correspondence matcher with ROI-aware aggregation
    
    Features:
    - Dense matching using LoFTR
    - ROI-aware weighted aggregation for point prediction
    - Can be used both as standard matcher and for direct point mapping
    """
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        super().__init__(config, device)
        
        # Try both config paths for backward compatibility
        matcher_config = config.get('matcher', {}).get('efficient_loftr', {})
        if not matcher_config:
            matcher_config = config.get('e2e_model', {}).get('efficient_loftr', {})
        
        # Load LoFTR backbone
        try:
            from kornia.feature import LoFTR as KorniaLoFTR
            
            model_type = matcher_config.get('model_type', 'outdoor')
            self.match_threshold = matcher_config.get('match_threshold', 0.2)
            
            pretrained_type = 'outdoor' if model_type == 'outdoor' else 'indoor'
            
            self.loftr = KorniaLoFTR(pretrained=pretrained_type).to(self.device).eval()
            self.available = True
            
            print(f"✓ Loaded EfficientLoFTR ({pretrained_type})")
            
        except ImportError:
            print("Warning: kornia not available, EfficientLoFTR disabled")
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
        Standard matching interface - returns dense correspondences
        
        Args:
            image0: First image
            image1: Second image
            keypoints0: Not used (LoFTR is detector-free)
            keypoints1: Not used
            descriptors0: Not used
            descriptors1: Not used
            
        Returns:
            Dictionary with:
                - keypoints0: Matched points in image0 (N, 2)
                - keypoints1: Matched points in image1 (N, 2)
                - confidence: Match confidence scores (N,)
        """
        if not self.available:
            return {
                'keypoints0': np.zeros((0, 2), dtype=np.float32),
                'keypoints1': np.zeros((0, 2), dtype=np.float32),
                'confidence': np.zeros((0,), dtype=np.float32)
            }
        
        # Get dense correspondences
        correspondences = self._get_correspondences(image0, image1)
        
        mkpts0 = correspondences['keypoints0']
        mkpts1 = correspondences['keypoints1']
        confidence = correspondences['confidence']
        
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
    
    def predict(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        roi_center: Optional[Tuple[int, int]] = None,
        roi_radius: Optional[int] = None
    ) -> Dict:
        """
        ROI-aware prediction for direct point mapping (E2E approach)
        
        Strategy:
        1. Use LoFTR to get dense correspondences
        2. If ROI specified, find correspondences near ROI
        3. Aggregate using weighted averaging to get final prediction
        
        Args:
            source_image: Source image
            target_image: Target image
            roi_center: Optional ROI center in source image (x, y)
            roi_radius: Optional ROI radius
            
        Returns:
            Dictionary with:
                - coords: Predicted coordinates in target image (x, y)
                - confidence: Prediction confidence
                - dense_map: Optional dense correspondence data
        """
        if not self.available:
            # Fallback to center
            return {
                'coords': np.array([
                    target_image.shape[1] // 2,
                    target_image.shape[0] // 2
                ], dtype=np.float32),
                'confidence': 0.0
            }
        
        # Get dense correspondences
        correspondences = self._get_correspondences(source_image, target_image)
        
        mkpts0 = correspondences['keypoints0']
        mkpts1 = correspondences['keypoints1']
        confidence = correspondences['confidence']
        
        # Filter by confidence
        mask = confidence > self.match_threshold
        mkpts0 = mkpts0[mask]
        mkpts1 = mkpts1[mask]
        confidence = confidence[mask]
        
        if len(mkpts0) == 0:
            # No matches found
            return {
                'coords': np.array([
                    target_image.shape[1] // 2,
                    target_image.shape[0] // 2
                ], dtype=np.float32),
                'confidence': 0.0
            }
        
        # Strategy: Find matches near ROI and aggregate
        if roi_center is not None and roi_radius is not None:
            # Find matches within ROI
            roi_x, roi_y = roi_center
            distances = np.sqrt(
                (mkpts0[:, 0] - roi_x) ** 2 + 
                (mkpts0[:, 1] - roi_y) ** 2
            )
            
            # Weight by distance and confidence
            weights = confidence * np.exp(-distances / (2 * roi_radius ** 2))
            
            # Weighted average
            if weights.sum() > 0:
                weights = weights / weights.sum()
                predicted_coords = (mkpts1.T @ weights).astype(np.float32)
                predicted_confidence = float(weights.max())
            else:
                # Fallback to simple average
                predicted_coords = mkpts1.mean(axis=0).astype(np.float32)
                predicted_confidence = float(confidence.mean())
        else:
            # No ROI specified - use center of source image
            roi_x = source_image.shape[1] // 2
            roi_y = source_image.shape[0] // 2
            
            distances = np.sqrt(
                (mkpts0[:, 0] - roi_x) ** 2 + 
                (mkpts0[:, 1] - roi_y) ** 2
            )
            
            weights = confidence * np.exp(-distances / 100.0)
            
            if weights.sum() > 0:
                weights = weights / weights.sum()
                predicted_coords = (mkpts1.T @ weights).astype(np.float32)
                predicted_confidence = float(weights.max())
            else:
                predicted_coords = mkpts1.mean(axis=0).astype(np.float32)
                predicted_confidence = float(confidence.mean())
        
        return {
            'coords': predicted_coords,
            'confidence': predicted_confidence,
            'dense_map': {
                'mkpts0': mkpts0,
                'mkpts1': mkpts1,
                'confidence': confidence
            }
        }
    
    def _get_correspondences(self, image0: np.ndarray, image1: np.ndarray) -> Dict:
        """
        Internal method to get dense correspondences from LoFTR
        
        Args:
            image0: First image
            image1: Second image
            
        Returns:
            Dictionary with keypoints0, keypoints1, confidence
        """
        # Convert to grayscale tensors
        if len(image0.shape) == 3:
            gray0 = image0.mean(axis=2)
            gray1 = image1.mean(axis=2)
        else:
            gray0 = image0
            gray1 = image1
        
        # Normalize
        gray0 = gray0.astype(np.float32) / 255.0
        gray1 = gray1.astype(np.float32) / 255.0
        
        # To tensors
        tensor0 = torch.from_numpy(gray0).unsqueeze(0).unsqueeze(0).to(self.device)
        tensor1 = torch.from_numpy(gray1).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Get correspondences
        with torch.no_grad():
            input_dict = {
                'image0': tensor0,
                'image1': tensor1
            }
            correspondences = self.loftr(input_dict)
        
        mkpts0 = correspondences['keypoints0'].cpu().numpy()
        mkpts1 = correspondences['keypoints1'].cpu().numpy()
        confidence = correspondences['confidence'].cpu().numpy()
        
        return {
            'keypoints0': mkpts0,
            'keypoints1': mkpts1,
            'confidence': confidence
        }
    
    @property
    def name(self) -> str:
        return "EfficientLoFTR"
    
    @property
    def is_learned(self) -> bool:
        return True

