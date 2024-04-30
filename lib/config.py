import logging
from logging.handlers import RotatingFileHandler
import os

from lib.utils import load_env

# ! This file is used to store global variables that are used throughout the project

#? Environment Variables ========================

env = load_env(['DEBUG', 'LOGS_FOLDER'])
DEBUG = int(env['DEBUG'])
debug_bool = (DEBUG > 0)

#? Logging ========================

log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s')
name = "lib"

handler = RotatingFileHandler(f"{env['LOGS_FOLDER']}/{name}.log", mode='a', maxBytes=5*1024*1024)
handler.setFormatter(log_formatter)

logger = logging.getLogger(name)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)