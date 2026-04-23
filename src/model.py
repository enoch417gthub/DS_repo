"""
model.py
BloodCellCNN - Custom CNN built from scratch.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    One convolutional stage:
    Conv2d(3x3, padding=1) -> BatchNorm2d -> ReLU -> [optional MaxPool2d]
    
    bias=False because BatchNorm already learns a shift (beta).
    """
    def __init__(self, in_ch, out_ch, pool=True, spatial_dropout=0.0):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        
        if spatial_dropout > 0:
            # Dropout2d zeros entire feature maps (channels)
            # More effective than dropping individual pixels
            layers.append(nn.Dropout2d(spatial_dropout))
        
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class BloodCellCNN(nn.Module):
    """
    Custom CNN for blood cell classification.
    Built entirely from scratch. No pretrained weights.
    
    Input shape: (batch, 3, 224, 224)
    Output shape: (batch, num_classes) -- raw logits, NO softmax
    """
    
    def __init__(self, num_classes=4, base_filters=32, dropout_rate=0.4):
        super().__init__()
        
        f = base_filters
        
        # Feature extractor
        # Input: (B, 3, 224, 224)
        self.features = nn.Sequential(
            # Block 1: 3 -> f, spatial: 224 -> 112
            ConvBlock(3, f, pool=True),
            
            # Block 2: f -> 2f, spatial: 112 -> 56
            ConvBlock(f, 2*f, pool=True),
            
            # Block 3: 2f -> 4f, spatial: 56 -> 28
            ConvBlock(2*f, 4*f, pool=True),
            
            # Block 4: 4f -> 8f, spatial: 28 -> 14
            ConvBlock(4*f, 8*f, pool=True),
            
            # Block 5: 8f -> 8f, spatial: 14 -> 14 (no pool)
            # Adds depth without further spatial reduction
            ConvBlock(8*f, 8*f, pool=False, spatial_dropout=0.1),
        )
        # Output of features: (B, 8f, 14, 14) [f=32 -> 256 channels]
        
        # Global Average Pooling
        # Collapses spatial dims: (B, 256, 14, 14) -> (B, 256, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        
        # Classifier head
        n_feat = 8 * f  # 256 for f=32
        
        self.classifier = nn.Sequential(
            nn.Flatten(),  # (B, 256, 1, 1) -> (B, 256)
            nn.Linear(n_feat, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),  # 0.4
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),  # 0.2
            nn.Linear(128, num_classes),  # final scores
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Kaiming (He) initialisation for conv/linear, ones/zeros for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.features(x)   # (B, 3, 224, 224) -> (B, 256, 14, 14)
        x = self.gap(x)        # (B, 256, 14, 14) -> (B, 256, 1, 1)
        x = self.classifier(x) # (B, 256, 1, 1) -> (B, num_classes)
        return x  # raw logits


# Verification
if __name__ == "__main__":
    model = BloodCellCNN(num_classes=4, base_filters=32)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # Expected: (2, 4)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")  # ~1.4M