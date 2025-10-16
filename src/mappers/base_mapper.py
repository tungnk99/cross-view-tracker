"""
Base Mapper Class
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import numpy as np


class BaseMapper(ABC):
    """
    Abstract base class cho mappers
    
    Mapper nhận matched points và tạo transformation để ánh xạ
    bất kỳ điểm nào từ ảnh source → ảnh target
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.transformation = None  # Lưu transformation đã fit
    
    @abstractmethod
    def fit(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        confidence: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Fit transformation từ matched points
        
        Args:
            src_points: Source points (N, 2)
            dst_points: Destination points (N, 2)
            confidence: Match confidence scores (N,)
            
        Returns:
            Dictionary chứa:
                - transformation: Transformation matrix hoặc parameters
                - inlier_mask: Boolean mask (N,) - inliers
                - statistics: Dict với num_inliers, inlier_ratio, reprojection_error
        """
        pass
    
    @abstractmethod
    def map_point(self, point: np.ndarray) -> np.ndarray:
        """
        Ánh xạ một điểm từ source → target
        
        Args:
            point: Point (x, y) hoặc (2,) array
            
        Returns:
            Mapped point (x, y)
        """
        pass
    
    def map_points(self, points: np.ndarray) -> np.ndarray:
        """
        Ánh xạ nhiều điểm
        
        Args:
            points: Points (N, 2)
            
        Returns:
            Mapped points (N, 2)
        """
        return np.array([self.map_point(p) for p in points])
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tên của mapper"""
        pass
    
    @property
    @abstractmethod
    def is_learned(self) -> bool:
        """Mapper có phải DL-based không"""
        pass

