"""
End-to-End Pipeline - Dense Correspondence
Dense Correspondence Model → Direct Mapping

Note: For true E2E, use models like DKM, PDC-Net, GLU-Net
"""

import time
import numpy as np
from typing import Dict, Tuple, Optional

from .base_pipeline import BasePipeline
from ..matchers import EfficientLoFTRMatcher


class E2EPipeline(BasePipeline):
    """
    End-to-End Pipeline: Dense Correspondence → Direct Mapping
    
    Components:
    - Dense Models: DKM, PDC-Net, GLU-Net, RAFT
    
    Đặc điểm:
    - 1 bước duy nhất (no explicit geometric fitting)
    - Dense pixel-wise correspondence
    - End-to-end learning
    - Direct mapping without RANSAC
    - Có thể xử lý cross-view gap lớn
    
    Note: This currently uses a LoFTR-based approach for dense correspondence.
          For true E2E models, consider implementing DKM or PDC-Net.
    """
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        super().__init__(config, device)
        
        pipeline_config = config.get('pipeline', {}).get('e2e', {})
        
        # Initialize E2E model (using matcher with predict capability)
        model_type = pipeline_config.get('model', 'efficient_loftr')
        
        if model_type == 'efficient_loftr':
            self.model = EfficientLoFTRMatcher(config, device)
        else:
            raise ValueError(f"Unknown E2E model: {model_type}")
        
        print(f"✓ E2E Model: {self.model.name}")
        
        # Load pretrained weights if specified
        model_path = pipeline_config.get('model_path', None)
        if model_path:
            self.model.load_weights(model_path)
            print(f"✓ Loaded weights from {model_path}")
        
        print(f"✓ E2E Pipeline initialized")
    
    def process_frame(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        roi_center: Optional[Tuple[int, int]] = None,
        roi_radius: Optional[int] = None,
        return_debug_info: bool = False
    ) -> Dict:
        """
        Process frame pair through E2E pipeline
        
        Pipeline:
        1. Single E2E model directly predicts mapping from source → target
        """
        start_time = time.time()
        debug_info = {} if return_debug_info else None
        
        # Step 1: E2E Prediction
        prediction = self.model.predict(
            source_image,
            target_image,
            roi_center=roi_center,
            roi_radius=roi_radius
        )
        
        coords = prediction['coords']
        confidence = prediction['confidence']
        
        if return_debug_info:
            debug_info['prediction'] = prediction
            if 'dense_map' in prediction:
                debug_info['has_dense_map'] = True
            if 'uncertainty' in prediction:
                debug_info['uncertainty'] = prediction['uncertainty']
        
        # Validate coordinates
        valid = (
            0 <= coords[0] < target_image.shape[1] and
            0 <= coords[1] < target_image.shape[0] and
            confidence > 0.2
        )
        
        processing_time = time.time() - start_time
        self._update_stats(valid, processing_time)
        
        result = {
            'coords': coords,
            'confidence': confidence,
            'valid': valid,
            'processing_time': processing_time,
            'reason': 'Success' if valid else 'Low confidence or out of bounds'
        }
        
        if return_debug_info:
            result['debug_info'] = debug_info
        
        return result
    
    def reset(self):
        """Reset pipeline state"""
        self.stats = {
            'total_frames': 0,
            'valid_predictions': 0,
            'avg_processing_time': 0.0
        }
    
    @property
    def pipeline_type(self) -> str:
        return "e2e"

