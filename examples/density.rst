2D Density Estimation Example
=============================

.. code-block:: python

   import torch
   import numpy as np
   from ngs import NGSConfig, build_ngs
   from ngs.training import NGSTrainer, TrainerConfig
   from ngs.visualization import plot_3d_gaussian_means

   # Generate 2D toy dataset
   def make_toy_density(n_samples=10000):
       # Mixture of 4 Gaussians
       centers = np.array([[-3, -3], [3, -3], [-3, 3], [3, 3]])
       data = []
       for c in centers:
           data.append(np.random.randn(n_samples // 4, 2) * 0.5 + c)
       return torch.FloatTensor(np.vstack(data))

   # Config for density estimation
   cfg = NGSConfig(
       max_k=32,
       k_init=8,
       latent_dim=2,
       routing="monolithic_mahalanobis",
       parameter_storage="direct_adapter",
       topology_control="continuous_density",
       memory_management="dynamic_growth",
       split_threshold=0.1,
       prune_threshold=0.02,
   )

   model = build_ngs(input_dim=2, output_dim=2, config=cfg)
   
   # Simple NLL training
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
   
   data = make_toy_density(5000)
   
   for epoch in range(100):
       idx = torch.randperm(len(data))[:256]
       batch = data[idx]
       
       optimizer.zero_grad()
       out = model(batch)
       
       # NLL loss for density estimation
       nll = -out.logits.log_softmax(dim=-1).mean()
       nll.backward()
       optimizer.step()
       
       if epoch % 20 == 0:
           print(f"Epoch {epoch}: NLL={nll.item():.4f}, K={model.K}")

   # Visualize learned Gaussians
   plot_3d_gaussian_means(model, save_path="density_gaussians.html")