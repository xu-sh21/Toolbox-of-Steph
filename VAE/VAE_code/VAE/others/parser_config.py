'''Parse the parameters and settings from config file !!!'''
################################################
import yaml
################################################

# Parse the config file.
def parse_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        config_dict = {}
        config_dict['params'] = {}
        config_dict['settings'] = {}

        # Params.
        config_dict['params']['batch_size'] = int(config['params']['batch_size'])
        config_dict['params']['epoch_nums'] = int(config['params']['epoch_nums'])
        config_dict['params']['lr'] = float(config['params']['lr'])
        config_dict['params']['act_func'] = str(config['params']['act_func']).lower()

        config_dict['params']['encoder_channels'] = config['params']['encoder_channels']
        config_dict['params']['encoder_kernel_size'] = int(config['params']['encoder_kernel_size'])
        config_dict['params']['decoder_channels'] = config['params']['decoder_channels']
        config_dict['params']['decoder_kernel_size'] = int(config['params']['decoder_kernel_size'])
        config_dict['params']['latent_dim'] = int(config['params']['latent_dim'])

        # Settings.
        config_dict['settings']['is_train'] = bool(config['settings']['is_train'])
        config_dict['settings']['new_model'] = bool(config['settings']['new_model'])
        config_dict['settings']['train_dir'] = str(config['settings']['train_dir'])
        config_dict['settings']['inference_version'] = int(config['settings']['inference_version'])
        config_dict['settings']['result_dir'] = str(config['settings']['result_dir'])
        config_dict['settings']['dataset'] = str(config['settings']['dataset'])
        config_dict['settings']['model_name'] = str(config['settings']['model_name'])
        config_dict['settings']['num_inferences'] = int(config['settings']['num_inferences'])
        

    return config_dict