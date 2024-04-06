import requests
import pytest

from lib.utils import load_env, get, post

@pytest.fixture
def messages():
    return [{"role": "assistant", "content": "What is the meaning of life"},]

@pytest.fixture
def URL():
    env = load_env()
    return env['SERVER_DEV_ENPOINT'] + '/chat'

@pytest.fixture
def HEADERS():
    env = load_env()
    return {"X-API-KEY": env['X-API-KEY']}

def test_endpoint_chat_root(HEADERS):
    URL = load_env()['SERVER_DEV_ENPOINT']
    res = get(URL, headers=HEADERS)
    assert res.status_code == 200

@pytest.mark.external
def test_endpoint_chat_openai(messages, URL, HEADERS):
    data = {
        "messages": messages,
        "model": "gpt-3.5-turbo",
    }

    res = post(URL, headers=HEADERS, data=data)
    parsed = res.json()

    assert res.status_code == 200
    assert parsed['data'] is not None
    assert parsed['data']['text'] != ''

@pytest.mark.external
def test_endpoint_chat_anthropic(messages, URL, HEADERS):
    data = {
        "messages": messages,
        "model": "claude-3-haiku",
    }

    res = post(URL, headers=HEADERS, data=data)
    parsed = res.json()

    assert res.status_code == 200
    assert parsed['data'] is not None
    assert parsed['data']['text'] != ''

@pytest.mark.external
def test_endpoint_chat_cohere(messages, URL, HEADERS):
    data = {
        "messages": messages,
        "temperature": 1,
        "max_tokens": 1000,
        "model": "command-light",
    }

    res = post(URL, headers=HEADERS, data=data)
    parsed = res.json()

    assert res.status_code == 200
    assert parsed['data'] is not None
    assert parsed['data']['text'] != ''