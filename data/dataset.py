import os
import random
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


def resolve_train_view_dir(root: Path) -> Path:
    """Resolve the training image directory for raw or processed PairUAV data."""
    candidates = [
        root / "train" / "drone",
        root / "train_tour",
        root / "train",
    ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        f"Could not find a training image directory under {root}. "
        "Expected one of: train/drone, train_tour, train."
    )


class PairDataset(Dataset):
    """PairUAV dataset from University-Release drone image pairs.
    
    For HARP-Pose-Lite (frozen backbone), we generate pairs by sampling
    two random drone images from the same building.
    
    Args:
        root: Path to University-Release root
        max_pairs: Maximum number of pairs (for subsampling)
        buildings: List of building IDs to use (subset for train/val split)
        seed: Random seed for reproducibility
        is_val: Validation mode (fixed pair selection)
    """
    
    def __init__(
        self,
        root: str,
        max_pairs: int = 960_000,
        buildings: Optional[List[str]] = None,
        seed: int = 42,
        is_val: bool = False,
    ):
        self.root = Path(root)
        self.max_pairs = max_pairs
        self.seed = seed
        self.is_val = is_val
        
        # Build per-building image lists
        view_dir = resolve_train_view_dir(self.root)
        self.view_dir = view_dir
        if buildings is None:
            buildings = sorted(os.listdir(view_dir))
        
        self.buildings = buildings
        self.image_index: Dict[str, List[str]] = {}
        
        for bld in buildings:
            bld_dir = view_dir / bld
            if bld_dir.is_dir():
                imgs = sorted([f for f in os.listdir(bld_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                if imgs:
                    self.image_index[bld] = imgs
        
        # Generate pairs
        self.pairs = self._generate_pairs()
        
        # Data transforms
        if is_val:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                     saturation=0.2, hue=0.1),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        
        # Target image gets NO augmentation (reference frame)
        self.target_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        print(f"[PairDataset] {len(self.buildings)} buildings, {len(self.pairs)} pairs")
    
    def _generate_pairs(self) -> List[dict]:
        """Generate image pairs with pseudo-relative pose labels."""
        rng = random.Random(self.seed)
        pairs = []
        
        total_available = 0
        for bld, imgs in self.image_index.items():
            n_imgs = len(imgs)
            # All possible unique pairs within this building
            n_pairs = n_imgs * (n_imgs - 1) // 2
            total_available += n_pairs
        
        # For training, we cap at max_pairs (subsampling)
        # For val, we use a smaller fixed set
        cap = self.max_pairs
        
        for bld, imgs in self.image_index.items():
            n = len(imgs)
            if n < 2:
                continue
            
            # Generate all pairs (or sample if there are too many)
            possible = n * (n - 1) // 2
            if self.is_val or possible <= cap * possible // total_available:
                # Use all pairs for val, or proportionally for train
                for i in range(n):
                    for j in range(i + 1, n):
                        pairs.append({
                            'building': bld,
                            'source': imgs[i],
                            'target': imgs[j],
                        })
            else:
                # Subsample proportionally
                n_sample = max(1, int(possible * cap / total_available))
                for _ in range(n_sample):
                    i, j = rng.sample(range(n), 2)
                    if i > j:
                        i, j = j, i
                    pairs.append({
                        'building': bld,
                        'source': imgs[i],
                        'target': imgs[j],
                    })
        
        if not self.is_val and len(pairs) > cap:
            pairs = rng.sample(pairs, cap)
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        pair = self.pairs[idx]
        bld = pair['building']
        view_dir = self.view_dir
        
        # Load images
        src_path = view_dir / bld / pair['source']
        tgt_path = view_dir / bld / pair['target']
        
        src_img = Image.open(src_path).convert("RGB")
        tgt_img = Image.open(tgt_path).convert("RGB")
        
        # Apply transforms
        source = self.transform(src_img)
        target = self.target_transform(tgt_img)
        
        # For HARP-Pose-Lite, we generate pseudo-labels based on image indices.
        # In the full dataset, true relative poses would come from camera metadata.
        # Here we use a proxy: the difference in image indices correlates with pose change.
        src_idx = int(pair['source'].split('-')[1].split('.')[0])
        tgt_idx = int(pair['target'].split('-')[1].split('.')[0])
        
        # Compute the heading and distance for this pair.
        # University-1652 drone images are taken at 72 azimuth angles,
        # 3 altitude levels, and we need to extract the ground truth.
        # For now, we use the image naming convention:
        # image-01.jpeg through image-54.jpeg correspond to:
        # 9 different altitudes × 6 azimuth views per altitude = 54 views per building
        # The first 9 images share the same azimuth but differ in altitude.
        # We group by azimuth and compute relative heading between groups.
        
        total_imgs = len(self.image_index[bld])  # usually 54
        
        # The images are organized into 72 azimuth angles across 3 altitude levels
        # image-01 to image-09: same azimuth, different altitudes
        # image-10 to image-18: next azimuth, etc.
        # So 54 / 9 = 6... but U-1652 docs say 72 azimuth angles
        # Actually the 54 images are: 18 azimuth × 3 altitudes = 54
        # So: img_num-1 // 3 = azimuth index (0..17)
        
        src_azimuth = (src_idx - 1) // 3
        tgt_azimuth = (tgt_idx - 1) // 3
        
        # Relative heading in degrees
        # 18 azimuth angles = 360/18 = 20° apart
        heading = (tgt_azimuth - src_azimuth) * 20.0
        heading = ((heading + 180) % 360) - 180  # Wrap to [-180, 180]
        
        # Pseudo-distance: based on angular separation between views
        # Larger azimuth difference → wider baseline → larger apparent distance
        az_diff = abs(tgt_azimuth - src_azimuth)
        # Empirical estimate: each view ~200m radius orbit, arc length per 20° ≈ 70m
        distance = az_diff * 70.0 / 18.0  # max ~70m at opposite views
        distance = max(distance, 0.0)
        
        return source, target, {
            'heading': heading,       # degrees, [-180, 180]
            'distance': distance,     # metres, [0, 130]
            'building': bld,
            'source_name': pair['source'],
            'target_name': pair['target'],
        }
