"""
Brute Force Matcher (Classical)
"""

import cv2
import numpy as np
from typing import Dict, Optional
from .base_matcher import BaseMatcher


class BruteForceMatcher(BaseMatcher):
    """
    Brute Force matcher with ratio test
    Dùng cho descriptor-based matching (SIFT, ORB, etc.)
    """
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        super().__init__(config, device)
        
        matcher_config = config.get('matcher', {}).get('brute_force', {})
        
        self.ratio_threshold = matcher_config.get('ratio_threshold', 0.75)
        self.cross_check = matcher_config.get('cross_check', True)
        
        # BFMatcher sẽ được tạo dynamic dựa trên descriptor type
    
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
        Match descriptors using brute force + ratio test
        """
        if descriptors0 is None or descriptors1 is None:
            raise ValueError("BruteForceMatcher requires descriptors")
        
        if len(descriptors0) == 0 or len(descriptors1) == 0:
            return {
                'keypoints0': np.zeros((0, 2), dtype=np.float32),
                'keypoints1': np.zeros((0, 2), dtype=np.float32),
                'confidence': np.zeros((0,), dtype=np.float32)
            }
        
        # Determine norm type based on descriptor dtype
        if descriptors0.dtype == np.uint8:
            # Binary descriptor (ORB) - use Hamming
            norm_type = cv2.NORM_HAMMING
        else:
            # Float descriptor (SIFT) - use L2
            norm_type = cv2.NORM_L2
        
        # Create matcher
        bf = cv2.BFMatcher(norm_type, crossCheck=False)
        
        # KNN match (k=2 for ratio test)
        matches = bf.knnMatch(descriptors0, descriptors1, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)
            elif len(match_pair) == 1:
                # Only one match found
                good_matches.append(match_pair[0])
        
        if len(good_matches) == 0:
            return {
                'keypoints0': np.zeros((0, 2), dtype=np.float32),
                'keypoints1': np.zeros((0, 2), dtype=np.float32),
                'confidence': np.zeros((0,), dtype=np.float32)
            }
        
        # Extract matched keypoints
        matched_kpts0 = np.array([keypoints0[m.queryIdx] for m in good_matches], dtype=np.float32)
        matched_kpts1 = np.array([keypoints1[m.trainIdx] for m in good_matches], dtype=np.float32)
        
        # Compute confidence from distance
        distances = np.array([m.distance for m in good_matches], dtype=np.float32)
        
        # Normalize distances to [0, 1] confidence
        if norm_type == cv2.NORM_HAMMING:
            # Hamming distance range [0, 256]
            confidence = 1.0 - (distances / 256.0)
        else:
            # L2 distance - normalize by max distance
            max_dist = distances.max() if distances.max() > 0 else 1.0
            confidence = 1.0 - (distances / max_dist)
        
        confidence = np.clip(confidence, 0, 1)
        
        return {
            'keypoints0': matched_kpts0,
            'keypoints1': matched_kpts1,
            'confidence': confidence
        }
    
    @property
    def name(self) -> str:
        return "BruteForce"
    
    @property
    def is_learned(self) -> bool:
        return False

