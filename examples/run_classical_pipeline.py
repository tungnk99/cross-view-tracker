"""
Example: Classical Pipeline - SIFT + BruteForce + Homography

This example demonstrates the classical sparse feature-based pipeline:
- Extractor: SIFT (most reliable and commonly used)
- Matcher: BruteForce with Lowe's ratio test
- Mapper: Homography with RANSAC

Usage:
    python examples/run_classical_pipeline.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.pipelines import create_pipeline


def visualize_results(source_img, target_img, result, roi_center=None):
    """Visualize the mapping results"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Source image with ROI
    axes[0].imshow(cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Source Image (Capitol View 1)', fontsize=14, fontweight='bold')
    
    if roi_center:
        # Draw ROI center point
        axes[0].plot(roi_center[0], roi_center[1], 'r*', markersize=20, label='ROI Center')
        # Draw circle around ROI
        circle = plt.Circle(roi_center, 50, color='red', fill=False, linewidth=2)
        axes[0].add_patch(circle)
    else:
        h, w = source_img.shape[:2]
        axes[0].plot(w//2, h//2, 'r*', markersize=20, label='Image Center')
    
    axes[0].legend()
    axes[0].axis('off')
    
    # Target image with mapped coordinates
    axes[1].imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Target Image (Capitol View 2)', fontsize=14, fontweight='bold')
    
    if result['valid']:
        mapped_coords = result['coords']
        axes[1].plot(mapped_coords[0], mapped_coords[1], 'g*', markersize=20, 
                    label=f"Mapped Point\nConfidence: {result['confidence']:.3f}")
        # Draw circle around mapped point
        circle = plt.Circle(mapped_coords, 50, color='green', fill=False, linewidth=2)
        axes[1].add_patch(circle)
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, f"Mapping Failed\n{result['reason']}", 
                    transform=axes[1].transAxes, ha='center', va='center',
                    fontsize=14, color='red', fontweight='bold')
    
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(project_root, 'examples', 'output_classical.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    plt.show()


def main():
    print("="*80)
    print("CLASSICAL PIPELINE EXAMPLE - SIFT + BruteForce + Homography")
    print("="*80)
    
    # Load images from data folder
    data_dir = os.path.join(project_root, 'data')
    source_path = os.path.join(data_dir, 'united_states_capitol_26757027_6717084061.jpg')
    target_path = os.path.join(data_dir, 'united_states_capitol_98169888_3347710852.jpg')
    
    print(f"\n📁 Loading images from data folder...")
    print(f"   Source: {os.path.basename(source_path)}")
    print(f"   Target: {os.path.basename(target_path)}")
    
    source_image = cv2.imread(source_path)
    target_image = cv2.imread(target_path)
    
    if source_image is None or target_image is None:
        print("❌ Error: Could not load images!")
        return
    
    print(f"✓ Images loaded successfully")
    print(f"   Source shape: {source_image.shape}")
    print(f"   Target shape: {target_image.shape}")
    
    # Create pipeline
    config_path = os.path.join(project_root, 'configs', 'pipeline_classical.yaml')
    print(f"\n📋 Loading config from: {os.path.basename(config_path)}")
    
    # Initialize pipeline with CPU (change to 'cuda' if you have GPU)
    device = 'cpu'
    pipeline = create_pipeline(config_path=config_path, device=device)
    
    print(f"\n{'='*80}")
    print("PROCESSING")
    print(f"{'='*80}")
    
    # Define ROI center (center of source image)
    h, w = source_image.shape[:2]
    roi_center = (w // 2, h // 2)
    print(f"\n📍 ROI Center: {roi_center}")
    
    # Process the frame pair
    print(f"\n🔄 Running pipeline...")
    result = pipeline.process_frame(
        source_image=source_image,
        target_image=target_image,
        roi_center=roi_center,
        roi_radius=50,
        return_debug_info=True
    )
    
    # Print results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"\n✓ Processing completed in {result['processing_time']:.3f} seconds")
    print(f"\n📊 Status: {'✓ VALID' if result['valid'] else '✗ INVALID'}")
    print(f"   Reason: {result['reason']}")
    print(f"   Mapped Coordinates: ({result['coords'][0]:.1f}, {result['coords'][1]:.1f})")
    print(f"   Confidence: {result['confidence']:.3f}")
    
    # Debug info
    if 'debug_info' in result:
        debug = result['debug_info']
        print(f"\n🔍 Debug Information:")
        print(f"   Keypoints in Source: {debug.get('num_keypoints0', 'N/A')}")
        print(f"   Keypoints in Target: {debug.get('num_keypoints1', 'N/A')}")
        print(f"   Matches Found: {debug.get('num_matches', 'N/A')}")
        
        if 'mapper_stats' in debug:
            stats = debug['mapper_stats']
            print(f"   Inliers: {stats.get('num_inliers', 'N/A')}")
            print(f"   Inlier Ratio: {stats.get('inlier_ratio', 'N/A'):.3f}")
    
    # Pipeline statistics
    stats = pipeline.get_stats()
    print(f"\n📈 Pipeline Statistics:")
    print(f"   Total Frames: {stats['total_frames']}")
    print(f"   Valid Predictions: {stats['valid_predictions']}")
    print(f"   Success Rate: {stats['valid_predictions']/max(stats['total_frames'], 1)*100:.1f}%")
    print(f"   Avg Processing Time: {stats['avg_processing_time']:.3f}s")
    
    # Visualize
    print(f"\n{'='*80}")
    print("VISUALIZATION")
    print(f"{'='*80}")
    visualize_results(source_image, target_image, result, roi_center)
    
    print(f"\n{'='*80}")
    print("✓ Example completed successfully!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

