import os
import json
import importlib.resources as pkg_resources

def load_config(filename, user_config_dir=None):
    """
    Load config JSON file.
    Priority:
    1. If user_config_dir is provided and file exists there, load that.
    2. Otherwise load from package defaults.
    """
    # 1. Check user override
    if user_config_dir:
        user_config_path = os.path.join(user_config_dir, filename)
        if os.path.exists(user_config_path):
            with open(user_config_path, 'r') as f:
                return json.load(f)

    # 2. Fallback to package default
    with pkg_resources.files('data_gatherer.config').joinpath(filename).open('r') as f:
        return json.load(f)