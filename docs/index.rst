.. pyiqa documentation master file, created by
   sphinx-quickstart on Sun Oct  1 15:56:36 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyiqa's documentation!
=================================

``pyiqa`` is an image quality assessment toolbox built with Python and PyTorch.
It provides full-reference (FR), no-reference (NR), and task-specific metrics.
Implementations are calibrated against official MATLAB or Python references
where available.

Basic Information 
-------------------------

.. toctree::
   :maxdepth: 1

   installation 
   examples
   gmad
   ModelCard
   benchmark

API Tools and References
-------------------------

.. toctree::
   :maxdepth: 2

   api_entries
   metrics_implement
   training_tools
   Dataset_Preparation


Citation
==================================

If you find our codes helpful to your research, please consider to use the following citation:
::

   @misc{pyiqa,
     title={{IQA-PyTorch}: PyTorch Toolbox for Image Quality Assessment},
     author={Chaofeng Chen and Jiadi Mo},
     year={2022},
     howpublished = "[Online]. Available: \url{https://github.com/chaofengc/IQA-PyTorch}"
   }


Please also consider to cite our new work **TOPIQ** if it is useful to you:
::

   @misc{chen2023topiq,
         title={TOPIQ: A Top-down Approach from Semantics to Distortions for Image Quality Assessment}, 
         author={Chaofeng Chen and Jiadi Mo and Jingwen Hou and Haoning Wu and Liang Liao and Wenxiu Sun and Qiong Yan and Weisi Lin},
         year={2023},
         eprint={2308.03060},
         archivePrefix={arXiv},
         primaryClass={cs.CV}
   }

   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
