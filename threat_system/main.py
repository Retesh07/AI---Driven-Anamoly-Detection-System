#!/usr/bin/env python3
"""
Main entry point for threat detection system.

Usage:
    python main.py --video path/to/video.mp4 [--output path/to/output] [--gpu]
    python main.py --webcam [--gpu]
"""

import argparse
import sys
from pathlib import Path
from pipeline import ThreatDetectionPipeline


def main():
    parser = argparse.ArgumentParser(
        description='Real-time threat detection system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Process video file
  python main.py --video input.mp4 --output ./results
  
  # Real-time webcam
  python main.py --webcam
  
  # With custom thresholds
  python main.py --video input.mp4 --violence-threshold 0.70 --warning-threshold 0.50
        '''
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video', type=str, help='Input video file path')
    input_group.add_argument('--webcam', action='store_true', help='Use webcam input')
    
    # Output options
    parser.add_argument('--output', type=str, default='./results',
                       help='Output directory (default: ./results)')
    
    # Threshold options
    parser.add_argument('--violence-threshold', type=float, default=None,
                       help='Violence detection threshold (default: calibrated)')
    parser.add_argument('--warning-threshold', type=float, default=None,
                       help='Warning threshold (default: calibrated)')
    
    # GPU options
    parser.add_argument('--gpu', action='store_true', help='Use GPU (default: auto-detect)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU mode')
    
    # Other options
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # ===== Device Selection =====
    if args.cpu:
        device = 'cpu'
    elif args.gpu:
        device = 'cuda'
    else:
        device = 'cuda'  # Auto-detect, default to CUDA
    
    if args.verbose:
        print(f'[Main] Device: {device}')
        print(f'[Main] Project root: {Path(__file__).parent}')
    
    # ===== Initialize Pipeline =====
    try:
        pipeline = ThreatDetectionPipeline(
            project_root=Path(__file__).parent,
            device=device,
            verbose=args.verbose
        )
    except Exception as e:
        print(f'[ERROR] Failed to initialize pipeline: {e}')
        return 1
    
    # ===== Process Input =====
    try:
        if args.video:
            results = pipeline.process_video(
                video_path=args.video,
                output_dir=args.output,
                violence_threshold=args.violence_threshold,
                warning_threshold=args.warning_threshold,
                export_json=True
            )
            
            print(f'\\n[Success] Results saved to: {args.output}')
            return 0
        
        elif args.webcam:
            print('[Error] Webcam mode not yet implemented')
            print('[Hint] Modify pipeline.py to support live capture')
            return 1
    
    except KeyboardInterrupt:
        print('\\n[Interrupted] Processing stopped by user')
        return 1
    except Exception as e:
        print(f'[ERROR] Processing failed: {e}')
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
