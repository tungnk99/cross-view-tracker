"""
FLANN Matcher (Classical - Fast)
"""

import cv2
import numpy as np
from typing import Dict, Optional
from .base_matcher import BaseMatcher


class FLANNMatcher(BaseMatcher):
    """
    FLANN (Fast Library for Approximate Nearest Neighbors) matcher
    Nhanh hơn BruteForce, dùng cho large-scale matching
    """
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        super().__init__(config, device)
        
        matcher_config = config.get('matcher', {}).get('flann', {})
        
        self.ratio_threshold = matcher_config.get('ratio_threshold', 0.75)
        self.trees = matcher_config.get('trees', 5)
        self.checks = matcher_config.get('checks', 50)
    
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
        Match descriptors using FLANN + ratio test
        """
        if descriptors0 is None or descriptors1 is None:
            raise ValueError("FLANNMatcher requires descriptors")
        
        if len(descriptors0) == 0 or len(descriptors1) == 0:
            return {
                'keypoints0': np.zeros((0, 2), dtype=np.float32),
                'keypoints1': np.zeros((0, 2), dtype=np.float32),
                'confidence': np.zeros((0,), dtype=np.float32)
            }
        
        # Determine algorithm based on descriptor type
        if descriptors0.dtype == np.uint8:
            # Binary descriptor (ORB) - use LSH
            index_params = dict(
                algorithm=6,  # FLANN_INDEX_LSH
                table_number=6,
                key_size=12,
                multi_probe_level=1
            )
        else:
            # Float descriptor (SIFT) - use KDTree
            index_params = dict(
                algorithm=1,  # FLANN_INDEX_KDTREE
                trees=self.trees
            )
        
        search_params = dict(checks=self.checks)
        
        # Create FLANN matcher
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # KNN match
        try:
            matches = flann.knnMatch(descriptors0, descriptors1, k=2)
        except cv2.error:
            # Fallback if FLANN fails
            return {
                'keypoints0': np.zeros((0, 2), dtype=np.float32),
                'keypoints1': np.zeros((0, 2), dtype=np.float32),
                'confidence': np.zeros((0,), dtype=np.float32)
            }
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)
            elif len(match_pair) == 1:
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
        
        # Compute confidence
        distances = np.array([m.distance for m in good_matches], dtype=np.float32)
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
        return "FLANN"
    
    @property
    def is_learned(self) -> bool:
        return False

