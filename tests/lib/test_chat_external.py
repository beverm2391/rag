import pytest
import inspect

from lib.chat import Chat
from lib.utils import validate_stream

# !!! These tests make actual API calls, so they are slow and should be run sparingly with the `--external` flag
        
@pytest.fixture
def example_args():
    temperature = 0
    max_tokens = 1000
    system_prompt = "You are a helpful assistant"
    return [temperature, max_tokens, system_prompt]

@pytest.fixture
def example_kwargs():
    return {"persist" : True}

@pytest.fixture
def example_prompts(): return "What is the meaning of life?", "Why is the sky blue?",


@pytest.mark.external
def test_chat_openai(example_args, example_prompts, example_kwargs):
    example_prompt_1, example_prompt_2 = example_prompts

    chat = Chat.create("gpt-3.5-turbo", *example_args, **example_kwargs) # persist true because we check number of messages later
    res = chat.chat(example_prompt_1)
    text = res['data']['text']

    assert res is not None, "Chat response is None"
    assert text is not None, "Chat response text is None"

    #? second message ========================
    res = chat.chat(example_prompt_2)
    text = res['data']['text']

    assert res is not None, "Chat response is None"
    assert text is not None, "Chat response text is None"
    assert (l:=len(chat.messages)) == 5, f"Chat messages should be 5 (system, user, chatbot, user, chatbot), is {l}"

@pytest.mark.external
def test_chat_anthropic(example_args, example_prompts, example_kwargs):
    example_prompt_1, example_prompt_2 = example_prompts

    chat = Chat.create("claude-3-haiku", *example_args, **example_kwargs)
    res = chat.chat(example_prompt_1)
    text = res['data']['text']

    assert res is not None, "Chat response is None"
    assert text is not None, "Chat response text is None"

    #? second message ========================
    res = chat.chat(example_prompt_2)
    text = res['data']['text']

    assert res is not None, "Chat response is None"
    assert text is not None, "Chat response text is None"
    assert (l:=len(chat.messages)) == 4, f"Chat messages should be 4 (user, chatbot, user, chatbot), is {l}" # Claude doesn't have a system message


@pytest.mark.external
def test_chat_cohere(example_args, example_prompts, example_kwargs):
    example_prompt_1, example_prompt_2 = example_prompts

    chat = Chat.create("command-r", *example_args, **example_kwargs)
    res = chat.chat(example_prompt_1)
    text = res['data']['text']

    assert res is not None, "Chat response is None"
    assert text is not None, "Chat response text is None"

    #? second message ========================
    res = chat.chat(example_prompt_2)
    text = res['data']['text']

    assert res is not None, "Chat response is None"
    assert text is not None, "Chat response text is None"
    assert (l:=len(chat.messages)) == 5, f"Chat messages should be 5 (system, user, chatbot, user, chatbot), is {l}"


# ! Stream =================================================

@pytest.mark.external
def test_chat_openai_stream(example_args, example_prompts, example_kwargs):
    chat = Chat.create("gpt-3.5-turbo", *example_args, **example_kwargs)
    res = chat.chat_stream(example_prompts[0])
    validate_stream(res)

    res = chat.chat_stream(example_prompts[1])
    validate_stream(res)
    assert (l:=len(chat.messages)) == 5, f"Chat messages should be 5 (system, user, chatbot, user, chatbot), is {l}"

@pytest.mark.external
def test_chat_anthropic_stream(example_args, example_prompts, example_kwargs):
    chat = Chat.create("claude-3-haiku", *example_args, **example_kwargs)
    res = chat.chat_stream(example_prompts[0])
    validate_stream(res)

    res = chat.chat_stream(example_prompts[1])
    validate_stream(res)

    assert (l:=len(chat.messages)) == 4, f"Chat messages should be 4 (user, chatbot, user, chatbot), is {l}" # Claude doesn't have a system message

@pytest.mark.external
def test_chat_cohere_stream(example_args, example_prompts, example_kwargs):
    example_args = [example_args[0], 1000, example_args[2]] # ? The command-light model has a context_window of 4096 so we need to reduce the max_tokens
    chat = Chat.create("command-light", *example_args, **example_kwargs)
    res = chat.chat_stream(example_prompts[0])
    validate_stream(res)

    res = chat.chat_stream(example_prompts[1])
    validate_stream(res)

    assert (l:=len(chat.messages)) == 5, f"Chat messages should be 5 (system, user, chatbot, user, chatbot), is {l}"