from lib.chat import Chat, ChatABC, OpenAIChat, AnthropicChat, CohereChat
import pytest 

def test_chat_factory():
    temperature = 0
    max_tokens = 1000
    system_prompt = "You are a helpful assistant"

    chat_oai = Chat("gpt-3.5-turbo", temperature, max_tokens, system_prompt)
    assert issubclass(type(chat_oai), ChatABC), "Chat is not a subclass of ChatABC"
    assert isinstance(chat_oai, OpenAIChat), f"Chat is not an instance of OpenAIChat, is {type(chat_oai)}"

    chat_anthropic = Chat("claude-3-haiku", temperature, max_tokens, system_prompt)
    assert issubclass(type(chat_anthropic), ChatABC), "Chat is not a subclass of ChatABC"
    assert isinstance(chat_anthropic, AnthropicChat), f"Chat is not an instance of AnthropicChat, is {type(chat_anthropic)}"

    chat_cohere = Chat("command-r", temperature, max_tokens, system_prompt)
    assert issubclass(type(chat_cohere), ChatABC), "Chat is not a subclass of ChatABC"
    assert isinstance(chat_cohere, CohereChat), f"Chat is not an instance of CohereChat, is {type(chat_cohere)}"

    # invalid model
    with pytest.raises(Exception):
        chat = Chat("invalid model", temperature, max_tokens, system_prompt)    

def test_chat_openai():
    pass

def test_chat_openai_stream():
    pass

def test_chat_anthropic():
    pass

def test_chat_anthropic_stream():
    pass

def test_chat_cohere():
    pass

def test_chat_cohere_stream():
    pass