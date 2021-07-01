import os
import toml

# config saved dir
config_dir = 'config'


def load_config(config_file):

    config_path = os.path.join(config_dir, config_file+'.toml')
    
    if os.path.exists(config_path):
        print('read config file from:', config_path)
        return toml.load(config_path)
    else:
        default_config_path = os.path.join(config_dir, 'default_config.toml')
        print(config_path, 'not exist,', 'read default file from:', default_config_path)
        return toml.load(default_config_path)
    
def save_config(config, config_file):
    
    config_path = os.path.join(config_dir, config_file+'.toml')
    
    with open(config_path, 'w') as f:
        toml.dump(config, f)
