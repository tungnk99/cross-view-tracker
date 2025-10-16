"""
Semi-Dense Pipeline - Semi-Dense Matcher
Semi-Dense Matcher → Mapper

Reference: "Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed" 
           (CVPR 2024)
"""

import time
import numpy as np
from typing import Dict, Tuple, Optional

from .base_pipeline import BasePipeline
from ..matchers import LoFTRMatcher, LightGlueMatcher, SuperGlueMatcher
from ..mappers import HomographyMapper, AffineMapper, LearnedMapper


class SemiDensePipeline(BasePipeline):
    """
    Semi-Dense Pipeline: Semi-Dense Matcher → Mapper
    
    Components:
    - Matcher: LoFTR/Efficient LoFTR (semi-dense), LightGlue, SuperGlue
    - Mapper: Homography, Affine, hoặc Learned
    
    Đặc điểm:
    - Semi-dense correspondences (1000s of matches vs sparse 100s)
    - 2 bước (matcher + mapper)
    - DL-based detector-free matching
    - High accuracy with interpretable geometric fitting
    - Recommended for most cross-view tracking tasks
    
    Reference:
    - "LoFTR: Detector-Free Local Feature Matching with Transformers" (CVPR 2021)
    - "Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed" (CVPR 2024)
    """
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        super().__init__(config, device)
        
        pipeline_config = config.get('pipeline', {}).get('semi_dense', {})
        
        # Initialize matcher (DL-based)
        matcher_type = pipeline_config.get('matcher', 'loftr')
        if matcher_type == 'loftr':
            self.matcher = LoFTRMatcher(config, device)
        elif matcher_type == 'lightglue':
            self.matcher = LightGlueMatcher(config, device)
        elif matcher_type == 'superglue':
            self.matcher = SuperGlueMatcher(config, device)
        else:
            raise ValueError(f"Unknown matcher: {matcher_type}")
        
        print(f"✓ Matcher: {self.matcher.name}")
        
        # Initialize mapper
        mapper_type = pipeline_config.get('mapper', 'homography')
        if mapper_type == 'homography':
            self.mapper = HomographyMapper(config)
        elif mapper_type == 'affine':
            self.mapper = AffineMapper(config)
        elif mapper_type == 'learned':
            self.mapper = LearnedMapper(config)
        else:
            raise ValueError(f"Unknown mapper: {mapper_type}")
        
        print(f"✓ Mapper: {self.mapper.name}")
        print(f"✓ Semi-Dense Pipeline initialized")
    
    def process_frame(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        roi_center: Optional[Tuple[int, int]] = None,
        roi_radius: Optional[int] = None,
        return_debug_info: bool = False
    ) -> Dict:
        """
        Process frame pair through semi-dense pipeline
        
        Pipeline:
        1. Match features using DL-based matcher (LoFTR/LightGlue)
        2. Fit mapper with matched points
        3. Map ROI center to target image
        """
        start_time = time.time()
        debug_info = {} if return_debug_info else None
        
        # Step 1: Semi-Dense Matching (detector-free DL-based)
        matches = self.matcher.match(
            source_image,
            target_image
        )
        
        num_matches = len(matches['keypoints0'])
        
        if return_debug_info:
            debug_info['num_matches'] = num_matches
            debug_info['avg_match_confidence'] = matches['confidence'].mean() if num_matches > 0 else 0.0
        
        # Check if enough matches
        if num_matches < 4:
            processing_time = time.time() - start_time
            self._update_stats(False, processing_time)
            
            result = {
                'coords': np.array([target_image.shape[1] // 2, target_image.shape[0] // 2], dtype=np.float32),
                'confidence': 0.0,
                'valid': False,
                'processing_time': processing_time,
                'reason': 'Insufficient matches'
            }
            
            if return_debug_info:
                result['debug_info'] = debug_info
            
            return result
        
        # Step 2: Fit Mapper
        mapper_result = self.mapper.fit(
            matches['keypoints0'],
            matches['keypoints1'],
            matches['confidence']
        )
        
        transformation = mapper_result['transformation']
        stats = mapper_result['statistics']
        
        if return_debug_info:
            debug_info['mapper_stats'] = stats
        
        # Check if mapping is valid
        if transformation is None:
            processing_time = time.time() - start_time
            self._update_stats(False, processing_time)
            
            result = {
                'coords': np.array([target_image.shape[1] // 2, target_image.shape[0] // 2], dtype=np.float32),
                'confidence': 0.0,
                'valid': False,
                'processing_time': processing_time,
                'reason': 'Mapper fitting failed'
            }
            
            if return_debug_info:
                result['debug_info'] = debug_info
            
            return result
        
        # Step 3: Map ROI center
        if roi_center is not None:
            roi_point = np.array(roi_center, dtype=np.float32)
        else:
            # Use center of source image
            roi_point = np.array([
                source_image.shape[1] // 2,
                source_image.shape[0] // 2
            ], dtype=np.float32)
        
        # Map the point
        mapped_coords = self.mapper.map_point(roi_point)
        
        # Compute confidence (combine match confidence and mapper statistics)
        match_conf = matches['confidence'].mean()
        mapper_conf = stats['inlier_ratio']
        confidence = (match_conf + mapper_conf) / 2.0
        
        # Validate coordinates
        valid = (
            0 <= mapped_coords[0] < target_image.shape[1] and
            0 <= mapped_coords[1] < target_image.shape[0] and
            confidence > 0.2
        )
        
        processing_time = time.time() - start_time
        self._update_stats(valid, processing_time)
        
        result = {
            'coords': mapped_coords,
            'confidence': confidence,
            'valid': valid,
            'processing_time': processing_time,
            'reason': 'Success' if valid else 'Low confidence or out of bounds'
        }
        
        if return_debug_info:
            debug_info['roi_point'] = roi_point
            debug_info['mapped_coords'] = mapped_coords
            debug_info['match_confidence'] = match_conf
            debug_info['mapper_confidence'] = mapper_conf
            result['debug_info'] = debug_info
        
        return result
    
    def reset(self):
        """Reset pipeline state"""
        self.mapper.transformation = None
        self.stats = {
            'total_frames': 0,
            'valid_predictions': 0,
            'avg_processing_time': 0.0
        }
    
    @property
    def pipeline_type(self) -> str:
        return "semi_dense"

