"""
Base Matcher Class
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np
import torch


class BaseMatcher(ABC):
    """
    Abstract base class cho matchers
    """
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        """
        Args:
            config: Configuration dictionary
            device: Device to run on
        """
        self.config = config
        self.device = device
    
    @abstractmethod
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
        Match features giữa 2 ảnh
        
        Args:
            image0: First image (có thể cần cho DL-based matchers)
            image1: Second image
            keypoints0: Keypoints từ image0 (cho descriptor-based matching)
            keypoints1: Keypoints từ image1
            descriptors0: Descriptors từ image0
            descriptors1: Descriptors từ image1
            
        Returns:
            Dictionary chứa:
                - keypoints0: np.ndarray shape (M, 2) - matched points trong image0
                - keypoints1: np.ndarray shape (M, 2) - matched points trong image1
                - confidence: np.ndarray shape (M,) - match confidence scores
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tên của matcher"""
        pass
    
    @property
    @abstractmethod
    def is_learned(self) -> bool:
        """Matcher có phải DL-based không"""
        pass

