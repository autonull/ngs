import re
import os

with open('ngs/modules/routers.py', 'r') as f:
    content = f.read()

old_init = """                    self.mu[s, :n].copy_(z_init[idx[:n]])"""
new_init = """                    z_sub = self.subspace_projectors[s](z_init[idx[:n]])
                    self.mu[s, :n].copy_(z_sub)"""
if old_init in content:
    content = content.replace(old_init, new_init)

old_norm = """        weight_sum = flat_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        flat_weights = flat_weights / weight_sum"""
new_norm = """        # Normalize weights so each sample sums to 1
        weight_sum = flat_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        flat_weights = flat_weights / weight_sum"""
if old_norm in content:
    content = content.replace(old_norm, new_norm)

def replace_max(match):
    indent = match.group(1)
    return f"{indent}if topk_vals.size(-1) > 0:\n{indent}    topk_vals = topk_vals - topk_vals.max(dim=-1, keepdim=True).values"

content = re.sub(r'([ \t]+)if topk_vals\.size\(-1\) > 0:\n[ \t]*topk_vals = topk_vals - topk_vals\.max\(dim=-1, keepdim=True\)\.values', r'\1topk_vals = topk_vals - topk_vals.max(dim=-1, keepdim=True).values', content)
content = re.sub(r'([ \t]+)topk_vals = topk_vals - topk_vals\.max\(dim=-1, keepdim=True\)\.values', replace_max, content)

content = content.replace("getattr(self, f'level_mu')[l]", "getattr(self, f'level_{l}_mu')")
content = content.replace("getattr(self, f'level_{l}_mu')", "self.level_mu[l]")
content = content.replace("getattr(self, f'level_{l}_log_s')", "self.level_log_s[l]")
content = content.replace("getattr(self, f'level_{l}_log_alpha')", "self.level_log_alpha[l]")
content = content.replace("getattr(self, f'level_{l}_active_mask')", "getattr(self, f'level_{l}_active_mask')")

old_lsr_mask = """        # Filter inactive units by setting their scores to -inf
        scores = torch.where(self.active_mask.unsqueeze(0), scores,
                            torch.tensor(-1e8, device=scores.device))"""

new_lsr_mask = """        # Filter inactive units by setting their scores to -inf
        scores = torch.where(self.active_mask.unsqueeze(0)[:scores.size(0), :scores.size(1)], scores,
                            torch.tensor(-1e8, device=scores.device))"""
if old_lsr_mask in content:
    content = content.replace(old_lsr_mask, new_lsr_mask)

old_lsr_init = """    def initialize_units(self, k_init: int, z_init: torch.Tensor = None):
        \"\"\"Initialize first k_init units, optionally from data z_init.\"\"\"
        if z_init is not None and z_init.size(0) > 0:
            with torch.no_grad():
                # Allow wrapping around if z_init is smaller than k_init
                if z_init.size(0) >= k_init:
                    idx = torch.randperm(z_init.size(0))[:k_init]
                    self.mu[:k_init].copy_(z_init[idx])
                else:
                    # Repeat z_init if needed
                    repeats = -(-k_init // z_init.size(0))
                    z_repeated = z_init.repeat(repeats, 1)[:k_init]
                    self.mu[:k_init].copy_(z_repeated)
                self.log_s[:k_init].fill_(0.0)
                self.log_alpha[:k_init].fill_(0.0)
        else:
            with torch.no_grad():
                self.mu[:k_init].normal_(0, 1.0)
                self.log_s[:k_init].fill_(0.0)
                self.log_alpha[:k_init].fill_(0.0)
        self.active_mask[:k_init] = True"""

new_lsr_init = """    def initialize_units(self, k_init: int, z_init: torch.Tensor = None):
        \"\"\"LSH Initialization ignores data-dependent mu for buckets.\"\"\"
        self.active_mask[:k_init] = True"""
if old_lsr_init in content:
    content = content.replace(old_lsr_init, new_lsr_init)

old_hier_init = """    def initialize_units(self, k_init: int, z_init: torch.Tensor = None):
        per_level = -(-k_init // self.num_levels)
        if z_init is not None and z_init.size(0) > 0:
            with torch.no_grad():
                idx = torch.randperm(z_init.size(0))[:k_init]
                for l in range(self.num_levels):
                    mask = getattr(self, f'level_{l}_active_mask')
                    n = min(per_level, len(mask))
                    self.level_mu[l][:n].copy_(z_init[idx[:n]])
                    self.level_log_s[l][:n].fill_(0.0)
                    self.level_log_alpha[l][:n].fill_(0.0)
                    mask[:n] = True
        else:"""
new_hier_init = """    def initialize_units(self, k_init: int, z_init: torch.Tensor = None):
        per_level = -(-k_init // self.num_levels)
        if z_init is not None and z_init.size(0) > 0:
            with torch.no_grad():
                if z_init.size(0) >= k_init:
                    idx = torch.randperm(z_init.size(0))[:k_init]
                    z_chosen = z_init[idx]
                else:
                    repeats = -(-k_init // z_init.size(0))
                    z_chosen = z_init.repeat(repeats, 1)[:k_init]
                for l in range(self.num_levels):
                    mask = getattr(self, f'level_{l}_active_mask')
                    n = min(per_level, len(mask))
                    start_idx = min(l * per_level, len(z_chosen))
                    end_idx = min(start_idx + n, len(z_chosen))
                    actual_n = end_idx - start_idx
                    if actual_n > 0:
                        self.level_mu[l][:actual_n].copy_(z_chosen[start_idx:end_idx])
                        self.level_log_s[l][:actual_n].fill_(0.0)
                        self.level_log_alpha[l][:actual_n].fill_(0.0)
                        mask[:actual_n] = True
        else:"""
if old_hier_init in content:
    content = content.replace(old_hier_init, new_hier_init)

old_init_mono = """            with torch.no_grad():
                idx = torch.randperm(z_init.size(0))[:k_init]
                self.mu[:k_init].copy_(z_init[idx])"""
new_init_mono = """            with torch.no_grad():
                # Allow wrapping around if z_init is smaller than k_init
                if z_init.size(0) >= k_init:
                    idx = torch.randperm(z_init.size(0))[:k_init]
                    self.mu[:k_init].copy_(z_init[idx])
                else:
                    # Repeat z_init if needed
                    repeats = -(-k_init // z_init.size(0))
                    z_repeated = z_init.repeat(repeats, 1)[:k_init]
                    self.mu[:k_init].copy_(z_repeated)"""
if old_init_mono in content:
    content = content.replace(old_init_mono, new_init_mono)

with open('ngs/modules/routers.py', 'w') as f:
    f.write(content)

with open('ngs/modules/topology_managers.py', 'r') as f:
    content = f.read()

old_tree = """                # Track tree structure
                device = router.active_mask.device
                if self.tree_depth is None:
                    self.tree_depth = torch.zeros_like(router.active_mask, dtype=torch.long, device=device)
                    self.tree_parent = torch.full_like(router.active_mask, -1, dtype=torch.long, device=device)
                    self.tree_children = [[] for _ in range(router.max_k)]"""
new_tree = """                # Track tree structure
                device = router.active_mask.device
                if self.tree_depth is None or self.tree_depth.device != device:
                    self.tree_depth = torch.zeros_like(router.active_mask, dtype=torch.long, device=device)
                    self.tree_parent = torch.full_like(router.active_mask, -1, dtype=torch.long, device=device)
                    self.tree_children = [[] for _ in range(router.max_k)]"""
if old_tree in content:
    content = content.replace(old_tree, new_tree)

with open('ngs/modules/topology_managers.py', 'w') as f:
    f.write(content)

with open('experiments/free_energy_manager.py', 'r') as f:
    content = f.read()

old_oop = """class FreeEnergyManager(HeuristicManager):"""
new_oop = """class FreeEnergyManager(AutopoieticManager):"""
if old_oop in content:
    content = content.replace(old_oop, new_oop)

old_step = """        if spawn_flag:
            self._spawn_gaussian(router)
            self.spawn_history.append(len(self.free_energy_history)-1)
            actions.append("spawn")

        return actions"""
new_step = """        if spawn_flag:
            self._spawn_gaussian(router)
            self.spawn_history.append(len(self.free_energy_history)-1)
            actions.append("spawn")

        # Match BaseTopologyManager signature Tuple[int, int, int] (pruned, split, spawned)
        return (0, 1 if split_flag else 0, 1 if spawn_flag else 0)"""
if old_step in content:
    content = content.replace(old_step, new_step)

with open('experiments/free_energy_manager.py', 'w') as f:
    f.write(content)

with open('ngs/models/ngs.py', 'r') as f:
    content = f.read()

old_div = """                if eye.any():
                    min_dist = dist[eye].min()
                    if not torch.isnan(min_dist):
                        losses.append(-min_dist)
            if not losses:
                return torch.tensor(0.0, device=self.p_down.weight.device)
            return torch.stack(losses).mean()

        # Monolithic / GaussianAttention / UncertaintyAware: mu is (max_k, latent_dim)
        mu = router.mu[active_idx]
        dist = torch.cdist(mu, mu, p=2)
        mask = ~torch.eye(len(active_idx), dtype=torch.bool, device=mu.device)
        if not mask.any():
            return torch.tensor(0.0, device=self.p_down.weight.device)
        min_dist = dist[mask].min()
        return -min_dist if not torch.isnan(min_dist) else torch.tensor(0.0, device=self.p_down.weight.device)"""

new_div = """                if eye.any():
                    # Repel all close pairs (softmin) to provide smoother gradients
                    close_dists = dist[eye]
                    if len(close_dists) > 0:
                        soft_min = -torch.logsumexp(-close_dists, dim=0)
                        losses.append(-soft_min)
            if not losses:
                return torch.tensor(0.0, device=self.p_down.weight.device)
            return torch.stack(losses).mean()

        # Monolithic / GaussianAttention / UncertaintyAware: mu is (max_k, latent_dim)
        mu = router.mu[active_idx]
        dist = torch.cdist(mu, mu, p=2)
        mask = ~torch.eye(len(active_idx), dtype=torch.bool, device=mu.device)
        if not mask.any():
            return torch.tensor(0.0, device=self.p_down.weight.device)
        close_dists = dist[mask]
        soft_min = -torch.logsumexp(-close_dists, dim=0)
        return -soft_min"""
if old_div in content:
    content = content.replace(old_div, new_div)

old_bce = """        err = self.error_density / (self.error_density.max() + 1e-8)
        # Use sigmoid output matching for BCE (both in [0,1])
        return F.binary_cross_entropy(sig, err.detach())"""
new_bce = """        err = self.error_density / (self.error_density.max() + 1e-8)
        # Clamp targets slightly to avoid log(0) in BCE if implemented custom later
        err = err.clamp(min=1e-6, max=1.0 - 1e-6)
        # Use sigmoid output matching for BCE (both in [0,1])
        return F.binary_cross_entropy(sig, err.detach())"""
if old_bce in content:
    content = content.replace(old_bce, new_bce)

old_k = """    @property
    def K(self) -> int:
        \"\"\"Number of active units.\"\"\"
        if hasattr(self.router, 'active_mask'):
            return self.router.active_mask.sum().item()
        if hasattr(self.router, 'K'):
            return self.router.K
        return self.config.max_k"""

new_k = """    @property
    def K(self) -> int:
        \"\"\"Number of active units.\"\"\"
        if not self._router_initialized:
            return self._k_init
        if hasattr(self.router, 'active_mask'):
            return self.router.active_mask.sum().item()
        if hasattr(self.router, 'K'):
            return self.router.K
        return self.config.max_k"""
if old_k in content:
    content = content.replace(old_k, new_k)

with open('ngs/models/ngs.py', 'w') as f:
    f.write(content)

with open('ngs/training/trainer.py', 'r') as f:
    content = f.read()

old_device = "        self.device = config.device"
new_device = "        self.device = config.device if torch.cuda.is_available() else 'cpu'"
if old_device in content:
    content = content.replace(old_device, new_device)

old_cfg = "    device: str = 'cuda'"
new_cfg = "    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'"
if old_cfg in content:
    content = content.replace(old_cfg, new_cfg)

with open('ngs/training/trainer.py', 'w') as f:
    f.write(content)

files_to_fix = ['tests/test_continual.py', 'tests/test_trainer.py']
for file in files_to_fix:
    with open(file, 'r') as f:
        content = f.read()
    content = content.replace("trainer = NGSTrainer(model, trainer_config)", "trainer = NGSTrainer(model.to('cpu'), trainer_config)")
    content = content.replace("device='cuda'", "device='cpu'")
    content = content.replace(".to('cuda')", ".to('cpu')")
    content = content.replace("device = 'cuda' if torch.cuda.is_available() else 'cpu'", "device = 'cpu'")
    with open(file, 'w') as f:
        f.write(content)
