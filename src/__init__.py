"""
Cervical Cancer Classification with SE Attention
================================================

This package contains the source code for the paper:
" Cervical Cancer Classification with Squeeze-and-Excitation Attention"

Models:
    - ResNet-50 Baseline
    - ResNet-50 + SE Attention (layer4)
    - ResNet-50 + SE Attention (avgpool)

Dataset: SIPaKMeD (cervical cell images, 5 classes)

Usage:
    python -m src.preprocess  # Preprocess data
    python -m src.train       # Train models
"""

__version__ = "1.0.0"