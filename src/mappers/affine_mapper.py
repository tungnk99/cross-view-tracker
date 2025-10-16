"""
Affine Mapper (Classical)
"""

import cv2
import numpy as np
from typing import Dict, Optional
from .base_mapper import BaseMapper


class AffineMapper(BaseMapper):
    """
    Affine transformation mapper
    Ước lượng affine matrix (2x3) - ít tham số hơn homography
    Phù hợp khi không có perspective distortion mạnh
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        mapper_config = config.get('mapper', {}).get('affine', {})
        
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
        Fit affine transformation using RANSAC
        
        Args:
            src_points: Source points (N, 2)
            dst_points: Destination points (N, 2)
            confidence: Match confidence scores (N,)
            
        Returns:
            Dict with transformation, inlier_mask, statistics
        """
        if len(src_points) < 3:
            return {
                'transformation': None,
                'inlier_mask': np.zeros(len(src_points), dtype=bool),
                'statistics': {
                    'num_inliers': 0,
                    'inlier_ratio': 0.0,
                    'reprojection_error': float('inf')
                }
            }
        
        # Estimate affine with RANSAC
        A, inlier_mask = cv2.estimateAffine2D(
            src_points,
            dst_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            confidence=self.ransac_confidence,
            maxIters=self.ransac_max_iters
        )
        
        if A is None:
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
            src_homo = np.hstack([src_inliers, np.ones((len(src_inliers), 1))])
            projected = (A @ src_homo.T).T
            
            # Compute error
            errors = np.linalg.norm(projected - dst_inliers, axis=1)
            reprojection_error = errors.mean()
        else:
            reprojection_error = float('inf')
        
        # Check if affine is valid
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
        self.transformation = A
        
        return {
            'transformation': A,
            'inlier_mask': inlier_mask,
            'statistics': {
                'num_inliers': num_inliers,
                'inlier_ratio': inlier_ratio,
                'reprojection_error': reprojection_error
            }
        }
    
    def map_point(self, point: np.ndarray) -> np.ndarray:
        """
        Map a point using affine transformation
        
        Args:
            point: Point (x, y) or (2,) array
            
        Returns:
            Mapped point (x, y)
        """
        if self.transformation is None:
            raise ValueError("Mapper not fitted. Call fit() first.")
        
        # Convert to homogeneous coordinates
        pt_homo = np.array([point[0], point[1], 1.0], dtype=np.float32)
        
        # Transform
        mapped = self.transformation @ pt_homo
        
        return mapped
    
    def map_points(self, points: np.ndarray) -> np.ndarray:
        """
        Map multiple points using affine transformation
        
        Args:
            points: Points (N, 2)
            
        Returns:
            Mapped points (N, 2)
        """
        if self.transformation is None:
            raise ValueError("Mapper not fitted. Call fit() first.")
        
        # Convert to homogeneous coordinates
        points_homo = np.hstack([points, np.ones((len(points), 1))]).astype(np.float32)
        
        # Transform all points at once
        mapped = (self.transformation @ points_homo.T).T
        
        return mapped
    
    @property
    def name(self) -> str:
        return "Affine"
    
    @property
    def is_learned(self) -> bool:
        return False

