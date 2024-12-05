'''Define the main function for VAE!!!'''
#######################################################################
import argparse
from others.parser_config import parse_config
from load_model.data_loader import load_data
from load_model.model_loader import load_model
from vae_core.train_module import *
from vae_core.test_module import *
#######################################################################

def main(config_path):
    # Parse the config file.
    config_dict = parse_config(config_path)
    # Load data.
    X_train, X_val, X_test = load_data(config_dict) 

    if config_dict['settings']['is_train'] == True: # Training process.
        begin_training(config_dict)
        device, vae_model, loss_func, optimizer = load_model(config_dict)
        if config_dict['settings']['new_model'] is False:
            vae_model = load_old_model(config_dict)
        train_job(device, vae_model, loss_func, optimizer, config_dict, X_train, X_val, X_test)

    elif config_dict['settings']['is_train'] == False: # Test process.
        begin_test(config_dict)
        vae_model = load_old_model(config_dict)
        test_job(vae_model, config_dict, X_test)


# Parse the command line arguments (config.yaml)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.yaml',
	help='The path of config.yaml. Default: ./config.yaml')
    args = parser.parse_args()

    # Check the path of config.yaml.
    config_path = args.config
    if not os.path.exists(config_path):
        raise ValueError(f'Path of Config File Does Not Exist!!! Please Check!!!')

    return config_path


if __name__ == "__main__":
    config_path = parse_args()
    main(config_path)