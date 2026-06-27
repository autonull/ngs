import torch
from ngs.core.interfaces import NGSConfig, RoutingStrategy
from ngs.models.ngs import MultiLayerNGS, SharedRouterNGS

# Config for MLP Proj / Dense Residual
config = NGSConfig(
    latent_dim=64,
    max_k=32,
    top_k=8,
    k_init=8,
    routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
    use_mlp_projections=True,
    beta_residual=0.2,
    gamma_residual=0.2
)
x = torch.randn(2, 128)

# Test 1: MultiLayerNGS
print("Testing MultiLayerNGS...")
multi_layer = MultiLayerNGS(d_in=128, d_out=10, num_layers=4, configs=[config]*4)
out = multi_layer(x)
print(f"Logits shape: {out.logits.shape}")

# Test 2: SharedRouterNGS
print("Testing SharedRouterNGS...")
shared_router = SharedRouterNGS(d_in=128, d_out=10, num_layers=4, config=config)
# requires manual init since lazy init is on standard NGSModel
shared_router.router.initialize_units(config.k_init, torch.randn(8, config.latent_dim))
out2 = shared_router(x)
print(f"Logits shape: {out2.logits.shape}")

print("Success!")
