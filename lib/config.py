import logging
from logging.handlers import RotatingFileHandler
import os
from dotenv import load_dotenv

# ! This file is used to store global variables that are used throughout the project

#? Environment Variables ========================

# ! Environment Variables ========================
# ??? This has to go in here (not utils) because we import from this file into utils (which would cause a circular import)
def load_env(expected_vars: list = []):
    env = load_dotenv()
    assert env, "No .env file found"
    for var in expected_vars:
        assert os.getenv(var), f"Expected {var} in .env"
    return os.environ

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