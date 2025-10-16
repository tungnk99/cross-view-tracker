"""
Homography Mapper (Classical)
"""

import cv2
import numpy as np
from typing import Dict, Optional
from .base_mapper import BaseMapper


class HomographyMapper(BaseMapper):
    """
    Homography-based mapper
    Ước lượng homography matrix (3x3) để ánh xạ điểm từ source → target
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        mapper_config = config.get('mapper', {}).get('homography', {})
        
        # RANSAC parameters
        self.ransac_threshold = mapper_config.get('ransac_threshold', 3.0)
        self.ransac_confidence = mapper_config.get('ransac_confidence', 0.995)
        self.ransac_max_iters = mapper_config.get('ransac_max_iters', 2000)
        self.min_inlier_ratio = mapper_config.get('min_inlier_ratio', 0.2)
    
    def fit(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        confidence: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Fit homography matrix using RANSAC
        
        Args:
            src_points: Source points (N, 2)
            dst_points: Destination points (N, 2)
            confidence: Match confidence scores (N,)
            
        Returns:
            Dict with transformation, inlier_mask, statistics
        """
        if len(src_points) < 4:
            return {
                'transformation': None,
                'inlier_mask': np.zeros(len(src_points), dtype=bool),
                'statistics': {
                    'num_inliers': 0,
                    'inlier_ratio': 0.0,
                    'reprojection_error': float('inf')
                }
            }
        
        # Estimate homography with RANSAC
        H, inlier_mask = cv2.findHomography(
            src_points,
            dst_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            confidence=self.ransac_confidence,
            maxIters=self.ransac_max_iters
        )
        
        if H is None:
            return {
                'transformation': None,
                'inlier_mask': np.zeros(len(src_points), dtype=bool),
                'statistics': {
                    'num_inliers': 0,
                    'inlier_ratio': 0.0,
                    'reprojection_error': float('inf')
                }
            }
        
        # Convert mask to boolean
        inlier_mask = inlier_mask.ravel().astype(bool)
        
        # Compute statistics
        num_inliers = inlier_mask.sum()
        inlier_ratio = num_inliers / len(src_points)
        
        # Compute reprojection error for inliers
        if num_inliers > 0:
            src_inliers = src_points[inlier_mask]
            dst_inliers = dst_points[inlier_mask]
            
            # Project source points
            projected = cv2.perspectiveTransform(
                src_inliers.reshape(-1, 1, 2),
                H
            ).reshape(-1, 2)
            
            # Compute error
            errors = np.linalg.norm(projected - dst_inliers, axis=1)
            reprojection_error = errors.mean()
        else:
            reprojection_error = float('inf')
        
        # Check if homography is valid
        if inlier_ratio < self.min_inlier_ratio:
            return {
                'transformation': None,
                'inlier_mask': inlier_mask,
                'statistics': {
                    'num_inliers': num_inliers,
                    'inlier_ratio': inlier_ratio,
                    'reprojection_error': reprojection_error
                }
            }
        
        # Save transformation
        self.transformation = H
        
        return {
            'transformation': H,
            'inlier_mask': inlier_mask,
            'statistics': {
                'num_inliers': num_inliers,
                'inlier_ratio': inlier_ratio,
                'reprojection_error': reprojection_error
            }
        }
    
    def map_point(self, point: np.ndarray) -> np.ndarray:
        """
        Map a point using homography
        
        Args:
            point: Point (x, y) or (2,) array
            
        Returns:
            Mapped point (x, y)
        """
        if self.transformation is None:
            raise ValueError("Mapper not fitted. Call fit() first.")
        
        # Reshape for cv2
        pt = np.array([[point]], dtype=np.float32)
        
        # Transform
        mapped = cv2.perspectiveTransform(pt, self.transformation)
        
        return mapped[0, 0]
    
    def map_points(self, points: np.ndarray) -> np.ndarray:
        """
        Map multiple points using homography
        
        Args:
            points: Points (N, 2)
            
        Returns:
            Mapped points (N, 2)
        """
        if self.transformation is None:
            raise ValueError("Mapper not fitted. Call fit() first.")
        
        # Transform all points at once
        mapped = cv2.perspectiveTransform(
            points.reshape(-1, 1, 2).astype(np.float32),
            self.transformation
        ).reshape(-1, 2)
        
        return mapped
    
    @property
    def name(self) -> str:
        return "Homography"
    
    @property
    def is_learned(self) -> bool:
        return False

