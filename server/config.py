import logging
from logging.handlers import RotatingFileHandler

from lib.utils import load_env
from lib.model_config import DEFAULTS, MODELS

# ! This file is used to store global variables that are used throughout the project

# Environment Variables
env = load_env(['DEBUG', 'SERVER_DEV_ENPOINT'])
DEBUG = int(env['DEBUG'])
debug_bool = (DEBUG > 0)

# Model Config
DEFAULTS = DEFAULTS
MODELS = MODELS

# Logging

log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s')
handler = RotatingFileHandler("logs/server.log", mode='a', maxBytes=5*1024*1024,)
handler.setFormatter(log_formatter)

logger = logging.getLogger("server")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)