# DFO-Tomographic Reconstruction

Advanced tomographic reconstruction algorithms using Dispersive Fly Optimization (DFO) with Total Variation (TV) regularization and hybrid SIRT-DFO methods for limited-angle computed tomography.

## Overview

This repository contains two Python implementations for solving inverse problems in computed tomography, specifically designed for limited-angle reconstruction scenarios where traditional methods struggle:

1. **DFO_DR_TV.py** - Dispersive Fly Optimization with Total Variation regularization
2. **HybridDFO_SIRT.py** - Hybrid approach combining SIRT initialization with DFO-TV refinement

## Key Features

- **Limited-Angle CT Reconstruction**: Optimized for scenarios with sparse projection angles (6 angles by default)
- **Total Variation Regularization**: Promotes edge-preserving smoothness in reconstructed images
- **Boolean Masking**: Incorporates a priori knowledge to constrain the solution space
- **ASTRA Toolbox Integration**: Efficient forward and backprojection operations
- **Convergence Visualization**: Automatic generation of convergence plots and quality metrics
- **Image Quality Metrics**: SNR and CNR computations for reconstruction evaluation

## Algorithms

### DFO with Total Variation (DFO_DR_TV.py)

Implements a nature-inspired optimization algorithm that:
- Uses a population of "flies" to explore the solution space
- Applies TV regularization (μ = 95.0) to maintain image smoothness
- Enforces pixel value bounds [0, 255]
- Applies boolean masking based on ray projection knowledge
- Optimizes the combined objective: `Fitness = e1 + μ × TV(x)`

**Key Parameters:**
- Population size: N=2
- Max function evaluations: 100,000
- TV regularization parameter: μ=95.0

### Hybrid SIRT-DFO (HybridDFO_SIRT.py)

A two-stage reconstruction approach:
1. **SIRT Initialization**: Runs Simultaneous Iterative Reconstruction Technique (20 iterations) to obtain a good initial estimate
2. **DFO-TV Refinement**: Further optimizes the SIRT result using DFO with TV regularization

This hybrid approach combines:
- The stability and speed of SIRT for initial reconstruction
- The refinement power of DFO for quality improvement
- TV regularization to maintain edge quality

## Requirements

```
numpy
imageio
Pillow
astra-toolbox
matplotlib
```

## Installation

```bash
pip install numpy imageio Pillow astra-toolbox matplotlib
```

## Usage

### Prepare Input Data
Place phantom images in an `input/` directory with the naming convention:
```
input/Phantom_0{phantomNo}_{size}x{size}.bmp
```

### Run DFO with TV
```bash
python DFO_DR_TV.py
```

### Run Hybrid SIRT-DFO
```bash
python HybridDFO_SIRT.py
```

### Configuration
Edit the parameters at the top of each script:
```python
size = 64           # Image size (64x64)
alpha = 6           # Number of projection angles
phantomNo = 5       # Phantom number to load
maxFE = 100000      # Maximum function evaluations
TV_MU = 95.0        # TV regularization parameter
```

## Output

Both scripts generate:
- Reconstructed images (BMP format)
- Boolean masking visualization
- Convergence plots showing:
  - Combined fitness (e1 + μ×TV)
  - Reconstruction error (e1)
  - Reproduction error (e2)
  - Total Variation values
- Performance metrics (SNR, CNR, execution time)

Output files are saved in:
- `recon_final1/` (DFO_DR_TV.py)
- `RARARA1/` (HybridDFO_SIRT.py)

## Technical Details

### Error Metrics
- **e1 (Reconstruction Error)**: Measures how well the reconstruction matches the sinogram data
- **e2 (Reproduction Error)**: Pixel-wise difference from ground truth phantom
- **TV (Total Variation)**: Measures image smoothness
- **SNR (Signal-to-Noise Ratio)**: μ_ROI / σ_ROI
- **CNR (Contrast-to-Noise Ratio)**: |μ_target - μ_background| / σ_background

### Boolean Table Masking
The algorithm uses a priori knowledge from specific projection angles to create a boolean mask that:
- Identifies pixels that cannot contain information based on zero-sum ray projections
- Reduces the search space for optimization
- Improves reconstruction quality in limited-angle scenarios

### Projection Geometry
- Parallel beam geometry
- Linear interpolation for forward/backprojection
- Angles evenly distributed over [0, π)

## Performance

Typical execution times (64×64 image, 6 angles):
- **DFO-TV only**: ~varies based on maxFE
- **Hybrid SIRT-DFO**: SIRT ~few seconds + DFO refinement

## Applications

This code is suitable for:
- Limited-angle CT reconstruction
- Sparse-view tomography
- Medical imaging with reduced radiation exposure
- Industrial non-destructive testing
- Research in inverse problems and optimization

## References

The implementation is based on research in:
- Dispersive Fly Optimization (DFO) algorithms
- Total Variation regularization for image reconstruction
- Simultaneous Iterative Reconstruction Technique (SIRT)
- Limited-angle tomographic reconstruction

## License

This project is available for academic and research purposes.

## Author

Developed for advanced tomographic reconstruction research.
