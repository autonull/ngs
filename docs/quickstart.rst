Quickstart
==========

Basic Usage
-----------

.. code-block:: python

   import torch
   from ngs import NGSConfig, build_ngs

   # Create configuration
   cfg = NGSConfig(
       max_k=64,
       k_init=16,
       latent_dim=32,
       routing="factorized_subspace",
       parameter_storage="hypernetwork_generated",
       topology_control="continuous_density",
       memory_management="dynamic_growth",
   )

   # Build model
   model = build_ngs(input_dim=784, output_dim=10, config=cfg)

   # Forward pass
   x = torch.randn(32, 784)
   output = model(x)
   print(f"Logits shape: {output.logits.shape}")
   print(f"Active units: {model.K}")

Training
--------

.. code-block:: python

   from ngs import NGSTrainer, TrainerConfig

   trainer_cfg = TrainerConfig(
       max_epochs=100,
       lr=1e-3,
       device="cuda",
   )

   trainer = NGSTrainer(model, trainer_cfg)
   trainer.fit(train_loader, val_loader)

Continual Learning
------------------

.. code-block:: python

   from ngs import NGSTrainer, TrainerConfig
   from ngs.core.interfaces import TopologyControl

   cfg = NGSConfig(
       max_k=256,
       k_init=64,
       topology_control=TopologyControl.CONTINUOUS_DENSITY,
       memory_management="strict_capacity",
   )

   model = build_ngs(784, 10, cfg)
   trainer = NGSTrainer(model, TrainerConfig(max_epochs=10))
   
   for task_id, (train_loader, val_loader) in enumerate(task_sequence):
       trainer.continual_fit(train_loader, val_loader, task_id)

Visualization
-------------

.. code-block:: python

   from ngs.visualization import plot_routing_heatmap, plot_3d_gaussian_means

   # Plot routing heatmap
   plot_routing_heatmap(model, save_path="routing.png")

   # Plot 3D Gaussian means
   plot_3d_gaussian_means(model, save_path="gaussians.html")