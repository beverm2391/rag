import pytest
import inspect

from lib.chat import Chat
from lib.utils import validate_stream

# !!! These tests make actual API calls, so they are slow and should be run sparingly
        
@pytest.fixture
def example_args():
    temperature = 0
    max_tokens = 1000
    system_prompt = "You are a helpful assistant"
    return [temperature, max_tokens, system_prompt]


@pytest.fixture
def example_prompt(): return "What is the meaning of life?"

@pytest.mark.external
def test_chat_openai(example_args, example_prompt):

    chat = Chat.create("gpt-3.5-turbo", *example_args)
    res = chat.chat(example_prompt)
    text = res['data']['text']

    assert res is not None, "Chat response is None"
    assert text is not None, "Chat response text is None"

@pytest.mark.external
def test_chat_anthropic(example_args, example_prompt):
    chat = Chat.create("claude-3-haiku", *example_args)
    res = chat.chat(example_prompt)
    text = res['data']['text']

    assert res is not None, "Chat response is None"
    assert text is not None, "Chat response text is None"

@pytest.mark.external
def test_chat_cohere(example_args, example_prompt):
    chat = Chat.create("command-r", *example_args)
    res = chat.chat(example_prompt)
    text = res['data']['text']

    assert res is not None, "Chat response is None"
    assert text is not None, "Chat response text is None"


# ! Stream =================================================

@pytest.mark.external
def test_chat_openai_stream(example_args, example_prompt):
    chat = Chat.create("gpt-3.5-turbo", *example_args)
    res = chat.chat_stream(example_prompt)
    validate_stream(res)

@pytest.mark.external
def test_chat_anthropic_stream(example_args, example_prompt):
    chat = Chat.create("claude-3-haiku", *example_args)
    res = chat.chat_stream(example_prompt)
    validate_stream(res)

@pytest.mark.external
def test_chat_cohere_stream(example_args, example_prompt):
    example_args = [example_args[0], 1000, example_args[2]] # ? The command-light model has a context_window of 4096 so we need to reduce the max_tokens
    chat = Chat.create("command-light", *example_args)
    res = chat.chat_stream(example_prompt)
    validate_stream(res)