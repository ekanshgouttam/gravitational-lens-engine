# Gravitational Lens Discovery Engine

A Vision Transformer-based tool for detecting gravitational lenses in astronomical survey images, with interpretable confidence scores built into the architecture via disentangled attention heads (arc geometry, Einstein ring, brightness asymmetry, colour gradient).

## Status
Version 1.0 in progress — environment setup complete.

## Environment
conda env create -f environment.yml
conda activate lensengine


## Tech Stack
PyTorch (CUDA 12.8), Zoobot (ConvNeXt-Nano pretrained backbone), lenstronomy, Astropy, Photutils, Astroquery
EOF