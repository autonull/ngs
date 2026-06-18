Architecture
============

Overview
--------

NGS (Neural Gaussian Systems) is a modular framework for adaptive neural networks that combines:

1. **Gaussian Mixture Models** in latent space for flexible density estimation
2. **Dynamic Topology Adaptation** - units can split, merge, spawn, and prune
3. **Modular Routing** - multiple strategies for selecting active units
4. **Parameter Sharing** - hypernetworks, LoRA, or direct adapters
5. **Riemannian Geometry** - for interpolating in parameter space

System Architecture
-------------------

.. mermaid::

   graph TD
       Input[Input x] --> Router[Router]
       Router -->|Top-K indices| ParamStore[Parameter Store]
       ParamStore -->|Adapters/Weights| Backbone[Backbone Network]
       Backbone --> Output[Output y]
       
       Router --> TopologyMgr[Topology Manager]
       TopologyMgr -->|Split/Merge/Spawn/Prune| Router
       TopologyMgr --> MemoryMgr[Memory Manager]
       MemoryMgr -->|Capacity/Buffers| Router
       MemoryMgr -->|Capacity/Buffers| ParamStore

Core Components
---------------

Router
~~~~~~

Six routing strategies implemented:

* **MonolithicMahalanobis** - Full covariance Mahalanobis distance
* **FactorizedSubspace** - Subspace projections with independent routing
* **LSHApproximate** - Locality-sensitive hashing for O(1) routing
* **Hierarchical** - Multi-level routing with coarse-to-fine selection
* **GaussianAttention** - Sparse attention over Gaussian units
* **UncertaintyAware** - Evidential routing with Dirichlet uncertainty

Parameter Store
~~~~~~~~~~~~~~~

Three parameter storage strategies:

* **DirectAdapter** - Full adapter matrices per unit
* **HypernetworkGenerated** - Weights generated from latent codes
* **LowRankAdapter** - LoRA-style low-rank decompositions

Topology Manager
~~~~~~~~~~~~~~~~

Four topology control strategies:

* **HeuristicManager** - Rule-based split/prune/spawn
* **ContinuousDensityManager** - Split-gate continuous relaxation
* **MergeAwareManager** - Cosine similarity merging with hysteresis
* **MetaLearnedManager** - RL-based topology actions

Memory Manager
~~~~~~~~~~~~~~

Three memory management strategies:

* **PreAllocatedManager** - Fixed buffer with masking
* **DynamicManager** - On-demand buffer expansion
* **StrictCapacityManager** - Hard capacity with LRU eviction

Data Flow
---------

1. Input :math:`x \in \mathbb{R}^{B \times D}` enters the router
2. Router computes assignments :math:`z \in \{0,1\}^{B \times K}` (top-k)
3. Parameter store generates/looks up adapters for active units
4. Backbone applies adapters to produce output
5. Topology manager monitors statistics and proposes changes
6. Memory manager enforces capacity constraints

Mathematical Formulation
------------------------

Routing (Monolithic)
~~~~~~~~~~~~~~~~~~~~

.. math::

   d_k(x) = (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k)

   \pi_k = \frac{\exp(-d_k / \tau)}{\sum_j \exp(-d_j / \tau)}

Routing (Factorized)
~~~~~~~~~~~~~~~~~~~~

.. math::

   d_k^{(m)}(x) = \|P_m x - \mu_k^{(m)}\|^2 / \sigma_k^{(m)2}

   \pi_k = \text{softmax}\left(-\sum_m d_k^{(m)} / \tau\right)

Split Gate (Continuous Density)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   g_k = \sigma\left(\frac{\rho_k - \tau_{\text{split}}}{\tau}\right)

   \mathcal{L}_{\text{split}} = -\sum_k g_k \log g_k + (1-g_k)\log(1-g_k)

Loss Functions
--------------

* **Task Loss**: Standard classification/regression loss
* **Entropy Loss**: Encourage diverse routing
* **Topology Loss**: Split/merge/spawn regularization
* **KD Loss**: Knowledge distillation for continual learning
* **Riemannian Loss**: Geodesic regularization in code space