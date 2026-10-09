Installation
=================

Requirements
------------
- Python 3.8 or later is recommended.
- PyTorch and torchvision must be compatible with each other. Install a build
  appropriate for your CPU/CUDA setup using the
  `official PyTorch instructions <https://pytorch.org/get-started/locally/>`_.
- A CUDA-enabled PyTorch installation and a supported NVIDIA GPU are required
  for GPU inference; CPU inference is also supported.
- Some optional metrics have additional model-specific dependencies or minimum
  library versions. See the corresponding model card.

Install with pip
----------------
::

    pip install pyiqa

Install the latest GitHub version
---------------------------------
::

    pip install git+https://github.com/chaofengc/IQA-PyTorch.git

Install a local checkout
------------------------
::

    git clone https://github.com/chaofengc/IQA-PyTorch.git
    cd IQA-PyTorch
    pip install -e .

The editable install includes the ``pyiqa`` and ``pyiqa-gmad`` command-line
entry points. To build a wheel and source distribution from a checkout, install
the ``build`` package and run ``python -m build`` from the repository root.
