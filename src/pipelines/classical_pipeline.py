"""
Classical Pipeline - Sparse Feature-Based
Extractor → Matcher → Mapper
"""

import time
import numpy as np
from typing import Dict, Tuple, Optional

from .base_pipeline import BasePipeline
from ..extractors import SIFTExtractor, ORBExtractor, SuperPointExtractor
from ..matchers import BruteForceMatcher, FLANNMatcher
from ..mappers import HomographyMapper, AffineMapper


class ClassicalPipeline(BasePipeline):
    """
    Classical Pipeline: Extractor → Matcher → Mapper
    
    Components:
    - Extractor: SIFT, ORB, SuperPoint
    - Matcher: BruteForce, FLANN
    - Mapper: Homography, Affine
    
    Đặc điểm:
    - Sparse features
    - Nhiều bước
    - Dễ debug, interpretable
    - Không cần training
    """
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        super().__init__(config, device)
        
        pipeline_config = config.get('pipeline', {}).get('classical', {})
        
        # Initialize extractor
        extractor_type = pipeline_config.get('extractor', 'sift')
        if extractor_type == 'sift':
            self.extractor = SIFTExtractor(config, device)
        elif extractor_type == 'orb':
            self.extractor = ORBExtractor(config, device)
        elif extractor_type == 'superpoint':
            self.extractor = SuperPointExtractor(config, device)
        else:
            raise ValueError(f"Unknown extractor: {extractor_type}")
        
        print(f"✓ Extractor: {self.extractor.name}")
        
        # Initialize matcher
        matcher_type = pipeline_config.get('matcher', 'brute_force')
        if matcher_type == 'brute_force':
            self.matcher = BruteForceMatcher(config, device)
        elif matcher_type == 'flann':
            self.matcher = FLANNMatcher(config, device)
        else:
            raise ValueError(f"Unknown matcher: {matcher_type}")
        
        print(f"✓ Matcher: {self.matcher.name}")
        
        # Initialize mapper
        mapper_type = pipeline_config.get('mapper', 'homography')
        if mapper_type == 'homography':
            self.mapper = HomographyMapper(config)
        elif mapper_type == 'affine':
            self.mapper = AffineMapper(config)
        else:
            raise ValueError(f"Unknown mapper: {mapper_type}")
        
        print(f"✓ Mapper: {self.mapper.name}")
        print(f"✓ Classical Pipeline initialized")
    
    def process_frame(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        roi_center: Optional[Tuple[int, int]] = None,
        roi_radius: Optional[int] = None,
        return_debug_info: bool = False
    ) -> Dict:
        """
        Process frame pair through classical pipeline
        
        Pipeline:
        1. Extract features from both images
        2. Match descriptors
        3. Fit mapper (homography/affine) with RANSAC
        4. Map ROI center to target image
        """
        start_time = time.time()
        debug_info = {} if return_debug_info else None
        
        # Step 1: Feature Extraction
        features0 = self.extractor.extract(source_image)
        features1 = self.extractor.extract(target_image)
        
        if return_debug_info:
            debug_info['num_keypoints0'] = len(features0['keypoints'])
            debug_info['num_keypoints1'] = len(features1['keypoints'])
        
        # Step 2: Matching
        matches = self.matcher.match(
            source_image,
            target_image,
            keypoints0=features0['keypoints'],
            keypoints1=features1['keypoints'],
            descriptors0=features0['descriptors'],
            descriptors1=features1['descriptors']
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
        
        # Step 3: Fit Mapper
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
        
        # Step 4: Map ROI center
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
        
        # Compute confidence from mapper statistics
        confidence = stats['inlier_ratio']
        
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
        return "classical"

