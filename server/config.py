from lib.utils import load_env
from lib.model_config import DEFAULTS, MODELS

# ! This file is used to store global variables that are used throughout the project

env = load_env(['DEBUG', 'SERVER_DEV_ENPOINT'])
debug = int(env['DEBUG']) > 0 # ? Boolean to check if the debug mode is on

DEFAULTS = DEFAULTS
MODELS = MODELS