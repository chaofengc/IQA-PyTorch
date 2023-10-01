Installation
=================

Dependencies
---------------
- Ubuntu >= 18.04
- Python >= 3.8
- PyTorch >= 1.12
- Torchvision >= 0.13
- CUDA >= 10.2 (if use GPU)

Install with pip
----------------
::

    pip install pyiqa

Install latest github version
----------------
::

    pip uninstall pyiqa # if have older version installed already 
    pip install git+https://github.com/chaofengc/IQA-PyTorch.git

Install with ``git clone``
----------------------
::

    git clone https://github.com/chaofengc/IQA-PyTorch.git
    cd IQA-PyTorch
    pip install -r requirements.txt
    python setup.py develop    
