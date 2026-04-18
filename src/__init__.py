"""
A Variance-Aware Benchmarking Workflow for Reproducible Deep Learning in Cervical Cytology
==========================================================================================

This package contains the source code for the paper:
"A Variance-Aware Benchmarking Workflow for Reproducible Deep Learning in Cervical Cytology:
 A Pilot Study on SIPaKMeD"

Models:
    - ResNet-50 Baseline
    - ResNet-50 + Channel Attention (layer4)
    - ResNet-50 + Channel Attention (avgpool)

Dataset: SIPaKMeD (cervical cell images, 5 classes)

Usage:
    python -m src.preprocess  # Preprocess data
    python -m src.train       # Train models
"""

__version__ = "1.0.0"