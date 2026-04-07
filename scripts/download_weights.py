#!/usr/bin/env python3
"""download_weights.py — Pre-download all model weights for fast GPU deployment.

This script downloads DINOv2-ResNet50 weights from timm/torchvision
and saves them locally so they're ready when training starts on a GPU machine.

Usage:
    python scripts/download_weights.py --out checkpoints/
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    # Import torch inside main to avoid import error if not installed yet
    try:
        import torch
    except ImportError:
        print("✗ PyTorch not installed. Install with: pip install torch")
        print("  Or create conda env first: conda env create -f environment.yml")
        sys.exit(1)
    parser = argparse.ArgumentParser(description='Pre-download model weights')
    parser.add_argument('--out', type=str, default='checkpoints/',
                       help='Output directory for weights')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Downloading weights to {args.out}...")
    print()
    
    all_ok = True
    
    # 1. timm ResNet50 (ImageNet-1k pretrained)
    print("[1/3] Downloading ResNet50 (timm, ImageNet-1k)...")
    try:
        import timm
        model = timm.create_model('resnet50', pretrained=True)
        # Trigger download by loading the model
        model.eval()
        save_path = os.path.join(args.out, 'resnet50_imagenet.pth')
        torch.save(model.state_dict(), save_path)
        size_mb = os.path.getsize(save_path) / 1e6
        print(f"  ✓ Saved to {save_path} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        all_ok = False
    
    # 2. DINOv2 ViT-B/14 (alternative backbone)
    print("\n[2/3] Checking DINOv2 availability...")
    try:
        import torch
        # Try torch.hub for DINOv2
        dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        dinov2_vitb14.eval()
        save_path = os.path.join(args.out, 'dinov2_vitb14.pth')
        torch.save(dinov2_vitb14.state_dict(), save_path)
        size_mb = os.path.getsize(save_path) / 1e6
        print(f"  ✓ Saved to {save_path} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  ✗ Failed (will download at training time): {e}")
    
    # 3. timm ResNet50.a1_in1k (DINO-pretrained variant)
    print("\n[3/3] Checking DINO-pretrained ResNet50 via timm...")
    try:
        import torch
        import timm
        # Try DINOv2 variant
        for model_name in ['resnet50.a1h_in1k', 'resnet50.a1_in1k', 'seresnet50.a1_in1k']:
            try:
                model = timm.create_model(model_name, pretrained=True)
                model.eval()
                save_path = os.path.join(args.out, f'{model_name}.pth')
                torch.save(model.state_dict(), save_path)
                size_mb = os.path.getsize(save_path) / 1e6
                print(f"  ✓ {model_name} → {save_path} ({size_mb:.1f} MB)")
                break
            except Exception:
                continue
        else:
            print("  ! No DINO-pretrained R50 variant found; standard ImageNet-1k will be used")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Summary
    print()
    weights = list(Path(args.out).glob('*.pth'))
    if weights:
        total_size = sum(w.stat().st_size for w in weights) / 1e6
        print(f"Downloaded {len(weights)} weight file(s), {total_size:.0f} MB total")
    else:
        print("No weights downloaded. They will be fetched automatically at training time.")
        
    if all_ok:
        print("\n✓ Ready for GPU deployment! Copy this directory to your GPU machine.")
    else:
        print("\n⚠ Some downloads failed. They'll be fetched at training time if possible.")


if __name__ == '__main__':
    main()