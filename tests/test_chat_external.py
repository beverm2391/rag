import pytest
import inspect

from lib.chat import Chat

# !!! These tests make actual API calls, so they are slow and should be run sparingly
        
@pytest.fixture
def example_args():
    temperature = 0
    max_tokens = 1000
    system_prompt = "You are a helpful assistant"
    return [temperature, max_tokens, system_prompt]

@pytest.fixture
def example_prompt(): return "What is the meaning of life?"


def test_chat_openai(example_args, example_prompt):

    chat = Chat("gpt-3.5-turbo", *example_args)
    res = chat.chat(example_prompt)
    text = res['data']['text']

    assert res is not None, "Chat response is None"
    assert text is not None, "Chat response text is None"


def test_chat_anthropic(example_args, example_prompt):
    chat = Chat("claude-3-haiku", *example_args)
    res = chat.chat(example_prompt)
    text = res['data']['text']

    assert res is not None, "Chat response is None"
    assert text is not None, "Chat response text is None"


def test_chat_cohere(example_args, example_prompt):
    chat = Chat("command-r", *example_args)
    res = chat.chat(example_prompt)
    text = res['data']['text']

    assert res is not None, "Chat response is None"
    assert text is not None, "Chat response text is None"


# ! Stream =================================================
    
def _validate_stream(res):
    """Helper function to validate a generator."""
    assert res is not None, "Chat response is None"
    assert inspect.isgenerator(res), "Chat response is not a generator"

    while True:
        try:
            next(res)
        except StopIteration:
            break
        except Exception as e:
            print(e)
            assert False, "Error in stream"
        finally:
            pass


def test_chat_openai_stream(example_args, example_prompt):
    chat = Chat("gpt-3.5-turbo", *example_args)
    res = chat.chat_stream(example_prompt)
    _validate_stream(res)

def test_chat_anthropic_stream(example_args, example_prompt):
    chat = Chat("claude-3-haiku", *example_args)
    res = chat.chat_stream(example_prompt)
    _validate_stream(res)


# def test_chat_cohere_stream(example_args, example_prompt):
#     chat = Chat("command-r", *example_args)
#     res = chat.chat_stream(example_prompt)
#     _validate_stream(res)