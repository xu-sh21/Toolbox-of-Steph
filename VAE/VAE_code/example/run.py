#########################
import sys
import subprocess
#########################

script_path = '../VAE/main.py'
config_option = '--config'
config_file = 'config.yaml'
command = ['python', script_path, config_option, config_file]

subprocess.run(command)