import json
import argparse
from omegaconf import OmegaConf

from src.visualize_and_save import visualize_and_save

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Getting models group you need to train")
    parser.add_argument("--config_path", type=str, default='config.yml', help="Path to the config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)

    visualize_and_save(config, 500, 20000)