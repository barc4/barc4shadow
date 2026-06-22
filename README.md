# barc4shadow

Small helper package for SHADOW4 beamlines.

It converts SHADOW4 beamlines into a simple layout dictionary and provides
Matplotlib plotting utilities for single or overlaid beamline configurations.

## Install

```bash
pip install barc4shadow
```

For local development:

```bash
pip install -e .
```

## Usage

```python
from barc4shadow import s4_beamline_to_layout, plot_beamline

layout = s4_beamline_to_layout(beamline)
plot_beamline(layout)
```

## Requirements

Python 3.10 or newer, with `numpy`, `matplotlib`, and `shadow4`.

---
[![PyPI](https://img.shields.io/pypi/v/barc4shadow.svg)]([https://pypi.org/project/barc4beams/](https://pypi.org/project/barc4shadow/))
[![License: CeCILL-2.1](https://img.shields.io/badge/license-CeCILL--2.1-blue.svg)](https://opensource.org/licenses/CECILL-2.1)
[![DOI](https://zenodo.org/badge/1263952255.svg)](https://doi.org/10.5281/zenodo.20734775)
