import json
import argparse
from omegaconf import OmegaConf

from src.models import make_default_model_use_layers_num
from src.models import make_default_model_use_kernel_size
from src.models import make_default_model_use_channel_size
from src.models import make_default_model_use_maxpool_position
from src.models import make_default_model_use_avgpool_position

from src.train_val import train_models_with_changed_params
from src.train_val import extract_models_results_with_changed_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Getting models group you need to train")
    parser.add_argument("--config_path", type=str,
                        help="Path to the config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    experiment_type = config.exp_type

    if experiment_type == 'change layers num':
        make_default_model_func = make_default_model_use_layers_num
        params_range = config.layers_num_range
    elif experiment_type == 'change kernel size':
        make_default_model_use_layers_num = make_default_model_use_kernel_size
        params_range = config.kernel_size_range
    elif experiment_type == 'change channels':
        make_default_model_func = make_default_model_use_channel_size
        params_range = config.channels_range
    elif experiment_type == 'change maxpool pos':
        make_default_model_func = make_default_model_use_maxpool_position
        params_range = config.maxpool_positions
    elif experiment_type == 'change avgpool pos':
        make_default_model_func = make_default_model_use_avgpool_position
        params_range = config.avgpool_positions


    train_models_with_changed_params(config,
                                    make_default_model_func,
                                    params_range)
    extract_models_results_with_changed_params(config,
                                               make_default_model_func,
                                               params_range)