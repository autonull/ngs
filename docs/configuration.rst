Configuration
=============

NGSConfig
---------

The ``NGSConfig`` dataclass controls all aspects of the model behavior.

.. autoclass:: ngs.core.interfaces.NGSConfig
   :members:
   :undoc-members:

Routing Strategies
------------------

.. autoclass:: ngs.core.interfaces.RoutingStrategy
   :members:

Parameter Storage Strategies
----------------------------

.. autoclass:: ngs.core.interfaces.ParameterStorage
   :members:

Topology Control Strategies
---------------------------

.. autoclass:: ngs.core.interfaces.TopologyControl
   :members:

Memory Management Strategies
----------------------------

.. autoclass:: ngs.core.interfaces.MemoryManagement
   :members:

Preset Configurations
---------------------

.. code-block:: python

   from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement

   # Lightweight config for few-shot learning
   fewshot_cfg = NGSConfig(
       max_k=64,
       k_init=16,
       latent_dim=32,
       routing=RoutingStrategy.FACTORIZED_SUBSPACE,
        parameter_storage=ParameterStorage.LORA,
        topology_control=TopologyControl.CONTINUOUS_DENSITY,
        memory_management=MemoryManagement.DYNAMIC,
   )

   # High-capacity config for continual learning
   continual_cfg = NGSConfig(
       max_k=512,
       k_init=128,
       latent_dim=64,
       routing=RoutingStrategy.HIERARCHICAL,
       parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
       topology_control=TopologyControl.META_LEARNED,
       memory_management=MemoryManagement.STRICT_CAPACITY,
   )

   # Uncertainty-aware config for safety-critical applications
   uncertainty_cfg = NGSConfig(
       max_k=128,
       k_init=32,
       latent_dim=64,
       routing=RoutingStrategy.UNCERTAINTY_AWARE,
       parameter_storage=ParameterStorage.DIRECT_ADAPTER,
       topology_control=TopologyControl.MERGE_AWARE,
        memory_management=MemoryManagement.PRE_ALLOCATED,
   )