import yaml
from pathlib import Path

def load_yaml(filename: str):
    """
    Load a YAML file from the config directory.

    :param filename: Name of the YAML file (e.g., 'country_codes.yaml').
    :return: Parsed content of the YAML file.
    """
    config_dir = Path(__file__).parent
    file_path = config_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Config file {filename} not found in {config_dir}")
    with open(file_path, "r") as file:
        return yaml.safe_load(file)
