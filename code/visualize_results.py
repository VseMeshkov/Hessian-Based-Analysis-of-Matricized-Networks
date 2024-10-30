import json
import argparse
from omegaconf import OmegaConf

from src.visualize_and_save import visualize_and_save

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Getting models group you need to train")
    parser.add_argument("--config_path", type=str, help="Path to the config file")
    parser.add_argument("--begin_with", type=str, default='10000',help="min sample size to visualize")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    begin_with = int(args.begin_with)

    visualize_and_save(config, 500, begin_with)
