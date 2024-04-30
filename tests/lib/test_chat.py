from lib.chat import Chat, ChatABC, OpenAIChat, AnthropicChat, CohereChat, MessageConverter
from typing import List, Dict
import pytest

# ! Fixtures ========================

@pytest.fixture
def example_universal():
    return [
        {'role': 'system', 'content': 'hello'},
        {'role': 'user', 'content': 'hi'},
        {'role': 'assistant', 'content': 'how can I help you?'},
        {'role': 'user', 'content': 'I need to buy a gun'},
        {'role': 'assistant', 'content': 'Okay! I can help you with that'},
    ]

# ! Message Converter ========================

# ? This test tests:
# - The conversion of a universal message to OpenAI format
# - The conversion of a universal message to Anthropic format
# - The conversion of a universal message to Cohere format

def test_message_converter_openai(example_universal: List[Dict]):
    example_openai = [
        {'role': 'system', 'content': 'hello'},
        {'role': 'user', 'content': 'hi'},
        {'role': 'assistant', 'content': 'how can I help you?'},
        {'role': 'user', 'content': 'I need to buy a gun'},
        {'role': 'assistant', 'content': 'Okay! I can help you with that'},
    ]
    assert MessageConverter.to_openai(example_universal).to_messages() == example_openai, "OpenAI message conversion failed"

def test_message_converter_anthropic(example_universal: List[Dict]):
    example_anthropic = [
        {'role': 'system', 'content': 'hello'},
        {'role': 'user', 'content': 'hi'},
        {'role': 'assistant', 'content': 'how can I help you?'},
        {'role': 'user', 'content': 'I need to buy a gun'},
        {'role': 'assistant', 'content': 'Okay! I can help you with that'},
    ]
    assert MessageConverter.to_anthropic(example_universal).to_messages() == example_anthropic, "Anthropic message conversion failed"

def test_message_converter_cohere(example_universal: List[Dict]):
    example_cohere = [
        {'role': 'SYSTEM', 'message': 'hello'},
        {'role': 'USER', 'message': 'hi'},
        {'role': 'CHATBOT', 'message': 'how can I help you?'},
        {'role': 'USER', 'message': 'I need to buy a gun'},
        {'role': 'CHATBOT', 'message': 'Okay! I can help you with that'},
    ]
    assert MessageConverter.to_cohere(example_universal).to_messages() == example_cohere, "Cohere message conversion failed"

# ! Chat Factory ========================

# ? This test tests:
# - The creation of a chat object with example parameters `Chat.create()`
# - The construction of the chat object with the correct model
# - The raising of an exception when an invalid model is passed
# - The raising of an exception when invalid messages are passed

def test_chat_factory(example_universal):
    temperature = 0
    max_tokens = 1000
    system_prompt = "You are a helpful assistant"

    # Test OpenAIChat ========================
    chat_oai = Chat.create(
        "gpt-3.5-turbo", temperature, max_tokens,
        system_prompt=system_prompt, messages=example_universal
    )
    assert issubclass(type(chat_oai), ChatABC), "Chat is not a subclass of ChatABC"
    assert isinstance(chat_oai, OpenAIChat), f"Chat is not an instance of OpenAIChat, is {type(chat_oai)}"

    # Test AnthropicChat ========================
    chat_anthropic = Chat.create(
        "claude-3-haiku", temperature, max_tokens,
        system_prompt=system_prompt, messages=example_universal
    )
    assert issubclass(type(chat_anthropic), ChatABC), "Chat is not a subclass of ChatABC"
    assert isinstance(chat_anthropic, AnthropicChat), f"Chat is not an instance of AnthropicChat, is {type(chat_anthropic)}"

    # Test CohereChat ========================
    chat_cohere = Chat.create(
        "command-r", temperature, max_tokens,
        system_prompt=system_prompt, messages=example_universal
        )
    assert issubclass(type(chat_cohere), ChatABC), "Chat is not a subclass of ChatABC"
    assert isinstance(chat_cohere, CohereChat), f"Chat is not an instance of CohereChat, is {type(chat_cohere)}"

    # Test invalid model ========================
    with pytest.raises(Exception):
        chat = Chat.create(
            "invalid model", temperature, max_tokens,
            system_prompt=system_prompt, messages=example_universal
        )

    # Test invalid messages ========================

    example_invalid_messages = [
        {'name': 'object', 'content': 'hello'},
        {'something' : 1},
    ]

    with pytest.raises(Exception):
        chat = Chat.create(
            "gpt-3.5-turbo", 100, max_tokens,
            system_prompt=system_prompt, messages=example_invalid_messages
        )