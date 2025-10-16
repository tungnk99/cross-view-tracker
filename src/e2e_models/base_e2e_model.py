"""
Base E2E Model Class
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import numpy as np
import torch


class BaseE2EModel(ABC):
    """
    Abstract base class cho End-to-End models
    
    E2E models nhận 2 ảnh và trực tiếp output ánh xạ pixel-to-pixel
    hoặc ROI mapping mà không cần extractor/matcher/mapper riêng biệt
    """
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        """
        Args:
            config: Configuration dictionary
            device: Device to run on
        """
        self.config = config
        self.device = device if torch.cuda.is_available() else 'cpu'
    
    @abstractmethod
    def predict(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        roi_center: Optional[Tuple[int, int]] = None,
        roi_radius: Optional[int] = None
    ) -> Dict:
        """
        Dự đoán ánh xạ từ source → target
        
        Args:
            source_image: Source image (H, W, 3) RGB
            target_image: Target image (H, W, 3) RGB
            roi_center: ROI center trong source image (x, y)
            roi_radius: ROI radius
            
        Returns:
            Dictionary chứa:
                - coords: Predicted coordinates (x, y) trong target image
                - confidence: Prediction confidence [0, 1]
                - dense_map: Optional dense correspondence map (H, W, 2)
                - uncertainty: Optional uncertainty estimation
        """
        pass
    
    @abstractmethod
    def load_weights(self, path: str):
        """Load pretrained weights"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tên của model"""
        pass

