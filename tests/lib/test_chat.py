from lib.chat import Chat, ChatABC, OpenAIChat, AnthropicChat, CohereChat
import pytest

# ! Chat Factory ========================
def test_chat_factory():
    temperature = 0
    max_tokens = 1000
    system_prompt = "You are a helpful assistant"

    chat_oai = Chat.create("gpt-3.5-turbo", temperature, max_tokens, system_prompt)
    assert issubclass(type(chat_oai), ChatABC), "Chat is not a subclass of ChatABC"
    assert isinstance(chat_oai, OpenAIChat), f"Chat is not an instance of OpenAIChat, is {type(chat_oai)}"

    chat_anthropic = Chat.create("claude-3-haiku", temperature, max_tokens, system_prompt)
    assert issubclass(type(chat_anthropic), ChatABC), "Chat is not a subclass of ChatABC"
    assert isinstance(chat_anthropic, AnthropicChat), f"Chat is not an instance of AnthropicChat, is {type(chat_anthropic)}"

    chat_cohere = Chat.create("command-r", temperature, max_tokens, system_prompt)
    assert issubclass(type(chat_cohere), ChatABC), "Chat is not a subclass of ChatABC"
    assert isinstance(chat_cohere, CohereChat), f"Chat is not an instance of CohereChat, is {type(chat_cohere)}"

    # invalid model
    with pytest.raises(Exception):
        chat = Chat.create("invalid model", temperature, max_tokens, system_prompt)

# ! Message Converter ========================

@pytest.fixture
def example_universal():
    return [
        {'role': 'system', 'content': 'hello'},
        {'role': 'user', 'content': 'hi'},
        {'role': 'assistant', 'content': 'how can I help you?'},
        {'role': 'user', 'content': 'I need to buy a gun'},
        {'role': 'assistant', 'content': 'Okay! I can help you with that'},
    ]

@pytest.fixture
def example_openai():
    return [
        {'role': 'system', 'content': 'hello'},
        {'role': 'user', 'content': 'hi'},
        {'role': 'assistant', 'content': 'how can I help you?'},
        {'role': 'user', 'content': 'I need to buy a gun'},
        {'role': 'assistant', 'content': 'Okay! I can help you with that'},
    ]

@pytest.fixture
def example_anthropic():
    return [
        {'role': 'system', 'content': 'hello'},
        {'role': 'user', 'content': 'hi'},
        {'role': 'assistant', 'content': 'how can I help you?'},
        {'role': 'user', 'content': 'I need to buy a gun'},
        {'role': 'assistant', 'content': 'Okay! I can help you with that'},
    ]

@pytest.fixture
def example_cohere():
    return [
        {'role': 'SYSTEM', 'message': 'hello'},
        {'role': 'USER', 'message': 'hi'},
        {'role': 'CHATBOT', 'message': 'how can I help you?'},
        {'role': 'USER', 'message': 'I need to buy a gun'},
        {'role': 'CHATBOT', 'message': 'Okay! I can help you with that'},
    ]

def test_message_converter_openai(example_universal, example_openai):
    assert Chat.convert_messages_to_openai(example_universal) == example_openai, "OpenAI message conversion failed"

def test_message_converter_anthropic(example_universal, example_anthropic):
    assert Chat.convert_messages_to_anthropic(example_universal) == example_anthropic, "Anthropic message conversion failed"

def test_message_converter_cohere(example_universal, example_cohere):
    assert Chat.convert_messages_to_cohere(example_universal) == example_cohere, "Cohere message conversion failed"