# SiamLight

This is a report on the paper "SiamLight: lightweight networks for object tracking via attention mechanisms and pixel-level cross-correlation", presented as a poster at the Journal of Real-Time Image Processing.

## Introduction

SiamLight aims to achieve fewer Flops and parameters. Our method has been extensively evaluated on multiple mainstream benchmarks and is found to produce stable performance improvements relative to the given baseline.

Please check out our project page, paper, dataset, and video for more details.

## Getting Started

### Installation

- Please refer to the installation instructions for PyTorch and PySOT in [INSTALL.md](http://install.md/).
- Add SiamLight to PYTHONPATH

### Test the Tracker

```
cd experiments

python -u ../../tools/test.py \\\\
--snapshot model.pth \\\\
--dataset OTB100 \\\\
--config config.yaml

```

### Evaluate the Tracker

```
python ../../tools/eval.py \\\\
--tracker_path ./results \\\\
--dataset OTB100 \\\\
--num 1 \\\\
--tracker_prefix 'model'

```

## Acknowledgements

- [PySOT](https://github.com/STVIR/pysot)
- [LightTrack](https://github.com/researchmm/LightTrack)