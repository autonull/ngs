import re

with open('ngs/modules/routers.py', 'r') as f:
    content = f.read()

# Fix LSRRouter initialization properly since python replacer failed earlier
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

content = content.replace(old_lsr_init, new_lsr_init)

with open('ngs/modules/routers.py', 'w') as f:
    f.write(content)
