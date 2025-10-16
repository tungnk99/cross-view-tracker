"""
Base Feature Extractor
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict
import numpy as np
import torch


class BaseExtractor(ABC):
    """
    Abstract base class cho feature extractors
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
    def extract(self, image: np.ndarray) -> Dict:
        """
        Trích xuất features từ ảnh
        
        Args:
            image: Input image (H, W, 3) RGB hoặc (H, W) grayscale
            
        Returns:
            Dictionary chứa:
                - keypoints: np.ndarray shape (N, 2) [x, y]
                - descriptors: np.ndarray shape (N, D)
                - scores: np.ndarray shape (N,) - confidence scores
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tên của extractor"""
        pass

