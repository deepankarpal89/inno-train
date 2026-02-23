import yaml
import sys
from pathlib import Path
from typing import Dict, Any


def read_yaml_config(file_path: str) -> Dict[str, Any]:
    """
    Read and parse a YAML configuration file.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        Dictionary containing the YAML configuration
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def print_config(config: Dict[str, Any], indent: int = 0) -> None:
    """
    Pretty print the configuration dictionary.
    
    Args:
        config: Configuration dictionary
        indent: Indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        elif isinstance(value, list):
            print("  " * indent + f"{key}:")
            for item in value:
                if isinstance(item, dict):
                    print_config(item, indent + 1)
                else:
                    print("  " * (indent + 1) + f"- {item}")
        else:
            print("  " * indent + f"{key}: {value}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python read_yaml_config.py <path_to_yaml_file>")
        print("\nExample:")
        print("  python scripts/read_yaml_config.py test_requests/entity_extraction_request_20260213_182013_b8555eb7-76ce-4aa5-81fe-cee6276058a6.yaml")
        sys.exit(1)
    
    yaml_file = sys.argv[1]
    
    try:
        print("=" * 70)
        print(f"Reading YAML configuration from: {yaml_file}")
        print("=" * 70)
        print()
        
        config = read_yaml_config(yaml_file)
        
        print_config(config)
        
        print()
        print("=" * 70)
        print("✅ Successfully read YAML configuration")
        print(f"📊 Total fields: {len(config)}")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ Error parsing YAML: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
