from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

from models.fusion import PairFusion
from models.heads import DistanceHead, DistanceHeadConfig, RotationHead


class SharedBackbone(nn.Module):
    """Backbone wrapper that exposes a spatial feature map and trainable stages."""

    def __init__(
        self,
        backbone_name: str = "auto",
        pretrained: bool = True,
        dino_weights_path: str | None = None,
    ) -> None:
        super().__init__()
        self.pretrained = pretrained
        self.requested_name = backbone_name
        self.backbone_name = self._resolve_backbone_name(backbone_name, dino_weights_path)
        self.used_dino_weights = False

        if self.backbone_name in {"dino_resnet50", "resnet50"}:
            self._family = "resnet"
            self._build_resnet(pretrained=pretrained, dino_weights_path=dino_weights_path)
        elif self.backbone_name == "convnext_tiny":
            self._family = "convnext"
            self._build_convnext(pretrained=pretrained)
        elif self.backbone_name == "efficientnet_v2_s":
            self._family = "efficientnet"
            self._build_efficientnet(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        self.current_mode = "full"

    @staticmethod
    def _resolve_backbone_name(backbone_name: str, dino_weights_path: str | None) -> str:
        normalized = backbone_name.lower().strip()
        if normalized not in {
            "auto",
            "dino_resnet50",
            "convnext_tiny",
            "efficientnet_v2_s",
            "resnet50",
        }:
            raise ValueError(f"Unknown backbone option: {backbone_name}")

        if normalized == "auto":
            if dino_weights_path and Path(dino_weights_path).is_file():
                return "dino_resnet50"
            return "convnext_tiny"

        if normalized == "dino_resnet50" and dino_weights_path and not Path(dino_weights_path).is_file():
            return "convnext_tiny"

        return normalized

    def _load_state_dict_flexible(self, model: nn.Module, weight_path: Path) -> bool:
        checkpoint = torch.load(str(weight_path), map_location="cpu")
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
        else:
            state_dict = checkpoint

        clean_state = {}
        for key, value in state_dict.items():
            clean_key = key
            for prefix in ("module.", "model.", "backbone."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
            clean_state[clean_key] = value

        missing, unexpected = model.load_state_dict(clean_state, strict=False)
        if len(unexpected) > 200:
            return False
        _ = missing
        return True

    def _resolve_dino_candidates(self, explicit_path: str | None) -> list[Path]:
        candidates: list[Path] = []
        if explicit_path:
            candidates.append(Path(explicit_path).expanduser())

        candidates.extend(
            [
                Path("checkpoints/dino_resnet50.pth"),
                Path("checkpoints/resnet50.a1h_in1k.pth"),
                Path("checkpoints/resnet50.a1_in1k.pth"),
            ]
        )
        unique: list[Path] = []
        seen: set[str] = set()
        for path in candidates:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            unique.append(path)
        return unique

    def _build_resnet(self, pretrained: bool, dino_weights_path: str | None) -> None:
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)

        if self.backbone_name == "dino_resnet50":
            for candidate in self._resolve_dino_candidates(dino_weights_path):
                if candidate.is_file() and self._load_state_dict_flexible(model, candidate):
                    self.used_dino_weights = True
                    break

        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.out_channels = 2048

    def _build_convnext(self, pretrained: bool) -> None:
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_tiny(weights=weights)
        self.features = model.features
        self.out_channels = model.classifier[2].in_features

    def _build_efficientnet(self, pretrained: bool) -> None:
        from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_v2_s(weights=weights)
        self.features = model.features
        self.out_channels = model.classifier[1].in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._family == "resnet":
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            return self.layer4(x)
        return self.features(x)

    def set_trainable(self, mode: str) -> None:
        normalized = mode.lower().strip()
        if normalized not in {"frozen", "upper", "full"}:
            raise ValueError(f"Invalid backbone mode: {mode}")

        for param in self.parameters():
            param.requires_grad = normalized == "full"

        if normalized == "upper":
            if self._family == "resnet":
                for module in (self.layer3, self.layer4):
                    for param in module.parameters():
                        param.requires_grad = True
            elif hasattr(self, "features"):
                feature_layers = list(self.features.children())
                for module in feature_layers[-2:]:
                    for param in module.parameters():
                        param.requires_grad = True

        self.current_mode = normalized


class GeoPairNet(nn.Module):
    """Shared trunk + specialist rotation and distance heads for PairUAV."""

    def __init__(
        self,
        backbone_name: str = "auto",
        pretrained: bool = True,
        dino_weights_path: str | None = None,
        global_dim: int = 1024,
        spatial_dim: int = 128,
        fused_dim: int = 1024,
        rotation_hidden_dim: int = 384,
        distance_hidden_dim: int = 384,
        distance_bins: int = 24,
        log_distance_min: float = 0.0,
        log_distance_max: float = 5.0,
        match_feature_dim: int = 8,
        geometry_feature_dim: int = 6,
        dropout: float = 0.1,
        use_uncertainty: bool = True,
        no_match_features: bool = False,
        no_geometry_features: bool = False,
        no_distance_bins: bool = False,
    ) -> None:
        super().__init__()

        self.match_feature_dim = match_feature_dim
        self.geometry_feature_dim = geometry_feature_dim
        self.global_dim = global_dim
        self.spatial_dim = spatial_dim
        self.no_match_features = no_match_features
        self.no_geometry_features = no_geometry_features
        self.no_distance_bins = no_distance_bins

        self.backbone = SharedBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            dino_weights_path=dino_weights_path,
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_projector = nn.Sequential(
            nn.Linear(self.backbone.out_channels, global_dim),
            nn.LayerNorm(global_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.spatial_projector = nn.Sequential(
            nn.Conv2d(self.backbone.out_channels, spatial_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(spatial_dim),
            nn.ReLU(inplace=True),
        )

        self.fusion = PairFusion(
            global_dim=global_dim,
            spatial_dim=spatial_dim,
            match_dim=match_feature_dim,
            geometry_dim=geometry_feature_dim,
            hidden_dim=fused_dim,
            fused_dim=fused_dim,
            dropout=dropout,
        )

        self.rotation_head = RotationHead(
            input_dim=fused_dim,
            hidden_dim=rotation_hidden_dim,
            dropout=dropout,
            with_uncertainty=use_uncertainty,
        )

        self.distance_head = DistanceHead(
            input_dim=fused_dim,
            hidden_dim=distance_hidden_dim,
            dropout=dropout,
            config=DistanceHeadConfig(
                log_distance_min=log_distance_min,
                log_distance_max=log_distance_max,
                num_bins=distance_bins,
            ),
            with_uncertainty=use_uncertainty,
        )

        self.backbone_mode = "full"

    def set_backbone_trainable(self, mode: str) -> None:
        self.backbone.set_trainable(mode)
        self.backbone_mode = mode

    def set_shared_projectors_trainable(self, trainable: bool) -> None:
        for module in (self.global_projector, self.spatial_projector):
            for param in module.parameters():
                param.requires_grad = trainable

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_raw = self.backbone(images)
        global_raw = self.global_pool(spatial_raw).flatten(1)

        global_embedding = self.global_projector(global_raw)
        spatial_embedding = self.spatial_projector(spatial_raw)

        return global_embedding, spatial_embedding

    def forward_from_embeddings(
        self,
        source_global: torch.Tensor,
        source_spatial: torch.Tensor,
        target_global: torch.Tensor,
        target_spatial: torch.Tensor,
        match_features: torch.Tensor | None = None,
        geometry_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.no_match_features:
            match_features = None
        if self.no_geometry_features:
            geometry_features = None

        fusion_outputs = self.fusion(
            source_global=source_global,
            target_global=target_global,
            source_spatial=source_spatial,
            target_spatial=target_spatial,
            match_features=match_features,
            geometry_features=geometry_features,
        )

        fused_features = fusion_outputs["fused_features"]
        rotation_outputs = self.rotation_head(fused_features)
        distance_outputs = self.distance_head(fused_features)

        output: dict[str, torch.Tensor] = {
            **fusion_outputs,
            **rotation_outputs,
            **distance_outputs,
            "heading": rotation_outputs["heading_deg"],
            "backbone_mode": torch.tensor(0.0, device=fused_features.device),
            "distance_bins_enabled": torch.tensor(
                0.0 if self.no_distance_bins else 1.0,
                device=fused_features.device,
            ),
        }
        return output

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        match_features: torch.Tensor | None = None,
        geometry_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        source_global, source_spatial = self.encode(source)
        target_global, target_spatial = self.encode(target)
        return self.forward_from_embeddings(
            source_global=source_global,
            source_spatial=source_spatial,
            target_global=target_global,
            target_spatial=target_spatial,
            match_features=match_features,
            geometry_features=geometry_features,
        )

    def summary(self) -> str:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        frozen = total - trainable
        text = (
            f"[GeoPairNet] backbone={self.backbone.backbone_name} "
            f"(dino_loaded={self.backbone.used_dino_weights}) "
            f"ablations(match={self.no_match_features}, geom={self.no_geometry_features}, "
            f"distance_bins={self.no_distance_bins}) "
            f"Total={total / 1e6:.2f}M Trainable={trainable / 1e6:.2f}M Frozen={frozen / 1e6:.2f}M"
        )
        print(text)
        return text

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (parameter for parameter in self.parameters() if parameter.requires_grad)
