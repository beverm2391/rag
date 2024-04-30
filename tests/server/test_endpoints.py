import requests

from lib.config import load_env

env = load_env(['DEBUG', 'SERVER_DEV_ENPOINT'])
DEV_ENDPOINT = env['SERVER_DEV_ENPOINT']

def _fetch(url: str):
    try:
        res = requests.get(url)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        return str(e)

def test_is_server_running():
    try:
        _fetch(DEV_ENDPOINT)
    except Exception as e:
        print("MAKE SURE THE SERVER IS RUNNING!")

def test_endpoints_root():
    _fetch(DEV_ENDPOINT + '/test')

def test_endpoints_example():
    _fetch(DEV_ENDPOINT + '/example')