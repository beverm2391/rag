import pytest
import inspect

from lib.chat import Chat
from lib.utils import validate_stream

# !!! These tests make actual API calls, so they are slow and should be run sparingly with the `--external` flag
        
@pytest.fixture
def example_args():
    temperature = 0
    max_tokens = 1000
    return [temperature, max_tokens]

@pytest.fixture
def example_kwargs():
    univeral_messages = [
        {'role': 'system', 'content': 'You are a helpful assistant'},
        {'role': 'user', 'content': 'hi'},
        {'role': 'assistant', 'content': 'how can I help you?'},
        {'role': 'user', 'content': 'I need to buy a gun'},
        {'role': 'assistant', 'content': 'Okay! I can help you with that'},
    ]

    return {
        "system_prompt" : "You are a helpful assistant",
        "messages" : univeral_messages
    }

@pytest.fixture
def prompts(): return "What is the meaning of life?", "Why is the sky blue?",

@pytest.mark.external
def test_chat_openai(prompts, example_args, example_kwargs):

    model = "gpt-3.5-turbo"

    # first we will text no system message and no messages
    chat = Chat.create(model, *example_args)
    res = chat.chat(prompts[0])
    text = res['data']['text']

    assert res is not None, "Chat response is None"
    assert text is not None, "Chat response text is None"

    # second we will test with a system message and messages
    chat = Chat.create(model, *example_args, **example_kwargs) # persist true because we check number of messages later
    res = chat.chat(prompts[0])
    text = res['data']['text']

    assert res is not None, "Chat response is None"
    assert text is not None, "Chat response text is None"

    # then we will test a second message on the same instance
    res = chat.chat(prompts[1])
    text = res['data']['text']

    assert res is not None, "Chat response is None"
    assert text is not None, "Chat response text is None"
    assert (l:=len(chat.messages)) == 9, f"Chat messages should be 9, is {l}"

@pytest.mark.external
def test_chat_anthropic(prompts, example_args, example_kwargs):

    model = "claude-3-haiku"

    # first we will text no system message and no messages
    chat = Chat.create(model, *example_args)
    res = chat.chat(prompts[0])
    text = res['data']['text']

    assert res is not None, "Chat response is None"
    assert text is not None, "Chat response text is None"

    # second we will test with a system message and messages
    chat = Chat.create(model, *example_args, **example_kwargs) # persist true because we check number of messages later
    res = chat.chat(prompts[0])
    text = res['data']['text']

    assert res is not None, "Chat response is None"
    assert text is not None, "Chat response text is None"

    # then we will test a second message on the same instance
    res = chat.chat(prompts[1])
    text = res['data']['text']

    assert res is not None, "Chat response is None"
    assert text is not None, "Chat response text is None"
    assert (l:=len(chat.messages)) == 8, f"Chat messages should be 8, is {l}"

@pytest.mark.external
def test_chat_cohere(prompts, example_args, example_kwargs):

    model = "command-r"

    # first we will text no system message and no messages
    chat = Chat.create(model, *example_args)
    res = chat.chat(prompts[0])
    text = res['data']['text']

    assert res is not None, "Chat response is None"
    assert text is not None, "Chat response text is None"

    # second we will test with a system message and messages
    chat = Chat.create(model, *example_args, **example_kwargs) # persist true because we check number of messages later
    res = chat.chat(prompts[0])
    text = res['data']['text']

    assert res is not None, "Chat response is None"
    assert text is not None, "Chat response text is None"

    # then we will test a second message on the same instance
    res = chat.chat(prompts[1])
    text = res['data']['text']

    assert res is not None, "Chat response is None"
    assert text is not None, "Chat response text is None"
    assert (l:=len(chat.messages)) == 9, f"Chat messages should be 9, is {l}"


# ! Stream =================================================

@pytest.mark.external
def test_chat_openai_stream(prompts, example_args, example_kwargs):
    model = "gpt-3.5-turbo"

    chat = Chat.create(model, *example_args, **example_kwargs)
    res = chat.chat_stream(prompts[0])
    validate_stream(res)
    assert (l:=len(chat.messages)) == 7, f"Chat messages should be 9, is {l}"

@pytest.mark.external
def test_chat_anthropic_stream(prompts, example_args, example_kwargs):
    model = "claude-3-haiku"

    chat = Chat.create(model, *example_args, **example_kwargs)
    res = chat.chat_stream(prompts[0])
    validate_stream(res)
    assert (l:=len(chat.messages)) == 6, f"Chat messages should be 8, is {l}"

@pytest.mark.external
def test_chat_cohere_stream(prompts, example_args, example_kwargs):
    model = "command-r"

    chat = Chat.create(model, *example_args, **example_kwargs)
    res = chat.chat_stream(prompts[0])
    validate_stream(res)
    assert (l:=len(chat.messages)) == 7, f"Chat messages should be 9, is {l}"