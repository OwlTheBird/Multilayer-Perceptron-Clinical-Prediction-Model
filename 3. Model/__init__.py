"""
Model package - provides clean imports for numbered modules.
This allows importing as 'config', 'dataset', etc. even though files are numbered.
"""

import importlib.util
from pathlib import Path

# Get the directory of this __init__.py file
_package_dir = Path(__file__).parent

# Import numbered modules and make them available with simple names
# This allows: import config, from dataset import, etc.

# Import config (01_config.py)
_config_spec = importlib.util.spec_from_file_location("config", _package_dir / "01_config.py")
config = importlib.util.module_from_spec(_config_spec)
_config_spec.loader.exec_module(config)

# Import dataset (02_dataset.py)
_dataset_spec = importlib.util.spec_from_file_location("dataset", _package_dir / "02_dataset.py")
dataset = importlib.util.module_from_spec(_dataset_spec)
_dataset_spec.loader.exec_module(dataset)

# Import model (03_model.py)
_model_spec = importlib.util.spec_from_file_location("model", _package_dir / "03_model.py")
model = importlib.util.module_from_spec(_model_spec)
_model_spec.loader.exec_module(model)

__all__ = ['config', 'dataset', 'model']

