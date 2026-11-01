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
    parser.add_argument('--camera-source', type=str, default='0',
                       help='Camera index (0, 1, ...) or RTSP/HTTP URL (default: 0)')
    
    # Output options
    parser.add_argument('--output', type=str, default='./results',
                       help='Output directory (default: ./results)')
    
    # Threshold options
    parser.add_argument('--violence-threshold', type=float, default=None,
                       help='Violence detection threshold (default: calibrated)')
    parser.add_argument('--warning-threshold', type=float, default=None,
                       help='Warning threshold (default: calibrated)')

    # Face identity options
    parser.add_argument('--face-db', type=str, default=None,
                       help='Folder containing enrolled family face images')
    
    # GPU options
    parser.add_argument('--gpu', action='store_true', help='Use GPU (default: auto-detect)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU mode')
    
    parser.add_argument('--dashboard', action='store_true', help='Serve the alert dashboard')
    parser.add_argument('--dashboard-host', default='127.0.0.1', help='Dashboard bind address (default: localhost)')
    parser.add_argument('--dashboard-port', type=int, default=8080, help='Dashboard HTTP port (default: 8080)')
    parser.add_argument('--headless', action='store_true', help='Run live inference without an OpenCV window; stop with Ctrl+C')

    # Other options
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    if args.headless and not args.webcam:
        parser.error('--headless requires --webcam')
    
    # ===== Device Selection =====
    if args.cpu:
        device = 'cpu'
    elif args.gpu:
        device = 'cuda'
    else:
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            # Initialization below produces the actionable dependency error.
            device = 'cpu'
    
    if args.verbose:
        print(f'[Main] Device: {device}')
        print(f'[Main] Project root: {Path(__file__).parent}')
    
    # ===== Initialize Pipeline =====
    try:
        # Delay heavyweight CV/model imports until after argument parsing.  This
        # keeps ``--help`` usable on deployment hosts before dependencies exist.
        from pipeline import ThreatDetectionPipeline
        pipeline = ThreatDetectionPipeline(
            project_root=Path(__file__).parent,
            device=device,
            verbose=args.verbose,
            face_db_path=args.face_db
        )
    except Exception as e:
        print(f'[ERROR] Failed to initialize pipeline: {e}')
        return 1
    
    # ===== Process Input =====
    dashboard = None
    try:
        if args.dashboard:
            from dashboard import Dashboard
            dashboard = Dashboard(source=args.video or args.camera_source)
            port = dashboard.start(args.dashboard_host, args.dashboard_port)
            print(f'[Dashboard] http://{args.dashboard_host}:{port}')
        if args.video:
            results = pipeline.process_video(
                video_path=args.video,
                output_dir=args.output,
                violence_threshold=args.violence_threshold,
                warning_threshold=args.warning_threshold,
                export_json=True,
                dashboard=dashboard
            )
            
            print(f'\\n[Success] Results saved to: {args.output}')
            return 0
        
        elif args.webcam:
            # Keep URLs as strings but make common camera indexes work without
            # requiring users to quote or otherwise special-case them.
            camera_source = int(args.camera_source) if args.camera_source.isdigit() else args.camera_source
            results = pipeline.process_realtime(
                camera_source=camera_source,
                output_dir=args.output,
                violence_threshold=args.violence_threshold,
                warning_threshold=args.warning_threshold,
                dashboard=dashboard,
                headless=args.headless,
            )
            print(f"\n[Success] Live session ended. Processed {results['processing_stats']['processed_frames']} frames.")
            return 0
    
    except KeyboardInterrupt:
        print('\\n[Interrupted] Processing stopped by user')
        return 1
    except Exception as e:
        print(f'[ERROR] Processing failed: {e}')
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if dashboard is not None:
            dashboard.stop()


if __name__ == '__main__':
    sys.exit(main())
