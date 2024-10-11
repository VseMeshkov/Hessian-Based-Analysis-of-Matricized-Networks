import json
import argparse
from omegaconf import OmegaConf

from src.models import make_default_model_use_layers_num
from src.train_val import train_models_with_changed_params
from src.train_val import extract_models_results_with_changed_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Getting models group you need to train")
    parser.add_argument("--config_path", type=str, default='config.yml', help="Path to the config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    experiment_type = config.exp_type
    if experiment_type == 'change layers num':
        train_models_with_changed_params(config,
                                         make_default_model_use_layers_num)
        extract_models_results_with_changed_params(config,
                                        make_default_model_use_layers_num)