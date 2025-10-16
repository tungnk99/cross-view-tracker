"""
Base Pipeline Class
Định nghĩa interface chung cho tất cả các pipeline
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import numpy as np
import torch


class BasePipeline(ABC):
    """
    Abstract base class cho tất cả các pipeline
    
    Mỗi pipeline phải implement:
    - __init__: Khởi tạo các components
    - process_frame: Xử lý một cặp ảnh
    - reset: Reset trạng thái internal
    """
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        """
        Khởi tạo pipeline
        
        Args:
            config: Configuration dictionary
            device: Device to run on ('cuda' or 'cpu')
        """
        self.config = config
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        if self.device == 'cpu' and device == 'cuda':
            print("Warning: CUDA not available, using CPU")
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'valid_predictions': 0,
            'avg_processing_time': 0.0
        }
    
    @abstractmethod
    def process_frame(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        roi_center: Optional[Tuple[int, int]] = None,
        roi_radius: Optional[int] = None,
        return_debug_info: bool = False
    ) -> Dict:
        """
        Xử lý một cặp ảnh
        
        Args:
            source_image: Ảnh nguồn (H, W, 3) RGB
            target_image: Ảnh đích (H, W, 3) RGB
            roi_center: Tâm ROI trong ảnh nguồn (x, y)
            roi_radius: Bán kính ROI
            return_debug_info: Trả về debug info hay không
            
        Returns:
            Dictionary chứa:
                - coords: Tọa độ ánh xạ (x, y) trong ảnh đích
                - confidence: Độ tin cậy [0, 1]
                - valid: Prediction có hợp lệ không
                - processing_time: Thời gian xử lý (giây)
                - debug_info: Debug information (nếu requested)
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset trạng thái pipeline"""
        pass
    
    def get_statistics(self) -> Dict:
        """Lấy thống kê pipeline"""
        total = self.stats['total_frames']
        if total == 0:
            return self.stats.copy()
        
        stats = self.stats.copy()
        stats['valid_rate'] = self.stats['valid_predictions'] / total
        stats['fps'] = 1.0 / self.stats['avg_processing_time'] if self.stats['avg_processing_time'] > 0 else 0
        
        return stats
    
    def _update_stats(self, is_valid: bool, processing_time: float):
        """Cập nhật statistics"""
        self.stats['total_frames'] += 1
        
        if is_valid:
            self.stats['valid_predictions'] += 1
        
        # Running average
        n = self.stats['total_frames']
        self.stats['avg_processing_time'] = (
            (self.stats['avg_processing_time'] * (n - 1) + processing_time) / n
        )
    
    @property
    @abstractmethod
    def pipeline_type(self) -> str:
        """Trả về loại pipeline: 'classical', 'semi-dense', 'e2e'"""
        pass

