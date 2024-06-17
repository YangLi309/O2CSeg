from mmcv import Config
from mmrazor.models.builder import build_algorithm
from mmcv.runner import build_optimizer

# Load the configuration file
cfg_path = 'path/to/your/config.py'  # Replace with your config file path
cfg = Config.fromfile(cfg_path)

# Build the model
model = build_algorithm(cfg.model)

# Build the optimizer
optimizer = build_optimizer(model, cfg.optimizer)

# Print the model parameters included in the optimizer
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.size())