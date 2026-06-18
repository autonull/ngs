Installation
============

Requirements
------------

* Python >= 3.9
* PyTorch >= 2.0
* CUDA >= 11.7 (for GPU support)

Install from Source
-------------------

.. code-block:: bash

   git clone https://github.com/yourusername/ngs.git
   cd ngs
   pip install -e .

Install with Optional Dependencies
----------------------------------

.. code-block:: bash

   # For visualization
   pip install -e .[viz]

   # For benchmarks
   pip install -e .[benchmarks]

   # For development
   pip install -e .[dev]

   # All extras
   pip install -e .[all]

Dependencies
------------

Core dependencies (in ``requirements.txt``):

* torch >= 2.0
* numpy >= 1.24
* pyyaml >= 6.0
* tqdm >= 4.65
* scipy >= 1.10

Visualization extras:

* matplotlib >= 3.7
* plotly >= 5.15
* dash >= 2.14

Benchmark extras:

* scikit-learn >= 1.3
* gymnasium >= 0.29
* torchvision >= 0.15