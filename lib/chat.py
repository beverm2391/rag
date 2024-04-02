from abc import ABC, abstractmethod
from typing import Generator
from openai import OpenAI
from anthropic import Anthropic
import cohere
import instructor

from lib.utils import MODELS, load_env

# TODO: The plan here is to have support for OpenAI, Anthropic, Cohere, and so on in one unified API
# This is so we can just plug the same code into any of these services and have it work
# Ideally we'll have a rag class that 

# chat class takex text in, and returns text out via a chat method
# expose the same chat methods (regular, streaming, etc) for all services
# Then a rag class can just take in an Index and a Chat class and work with any service

# lets make an abstract class for chat

env = load_env()
OPENAI_API_KEY = env.get("OPENAI_API_KEY", None)
ANTHROPIC_API_KEY = env.get("ANTHROPIC_API_KEY", None)
COHERE_API_KEY = env.get("COHERE_API_KEY", None)

class ChatABC(ABC):
    """Abstract base class for chat services."""
    def __init__(self, model_name: str, temperature: float = 0, max_tokens: int = 2048, system_prompt: str = "You are a helpful assistant.", debug: bool = False):
        self.model_name = model_name
        self.model_var = None
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.messages = []
        self.debug = debug

    @abstractmethod
    def _init_client(self):
        """Initialize the client for the chat service."""
        pass

    @abstractmethod
    def chat(self, text: str) -> str:
        """Chat with the model and return the response."""
        pass

    @abstractmethod
    def chat_stream(self, text: str) -> Generator[str, None, None]:
        """Chat with the model and yield the response."""
        pass

    def __repr__(self): return f"{self.__class__.__name__}(model={self.model})"

class OpenAIChat(ChatABC):
    """Chat with OpenAI"""
    _name = "openai"
    _api_key = OPENAI_API_KEY

    def __init__(self, model_name: str, temperature: float, max_tokens: int, system_prompt: str, debug: bool = False, model_var: str = None, context_window: int = 0):
        super().__init__(model_name)
        self._init_client()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.model_var = model_var | model_name
        self.context_window = context_window

        self.messages = [
            {"role": "system", "content": system_prompt},
        ]

        self.debug = debug
        
    def _init_client(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        if self.debug: print(f"Initialized {self._name} client.")

    def chat(self, text: str) -> str:
        res = self.client.chat.completions.create(
            model=self.model_var,
            messages=self.messages + [{"role": "user", "content": text}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self.messages.append(res.choices[0].message)
        return self.messages[-1]['content']

    def chat_stream(self, text: str) -> Generator[str, None, None]:
        res_stream = self.client.chat.completions.create(
            model=self.model_var,
            messages=self.messages + [{"role": "user", "content": text}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        new_message = {"role": "assistant", "content": ""}
        for res in res_stream:
            if res.choices[0].delta.get("content"):
                new_message["content"] += res.choices[0].delta["content"]
                yield new_message["content"]

        self.messages.append(new_message)

class AnthropicChat(ChatABC):
    """Chat with Anthropic"""
    _name = "anthropic"
    _api_key = ANTHROPIC_API_KEY

    def __init__(self, model_name: str, temperature: float, max_tokens: int, system_prompt: str, debug: bool = False, model_var: str = None, context_window: int = 0):
        super().__init__(model_name)
        self._init_client()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.model_var = model_var | model_name
        self.context_window = context_window

        self.messages = [
            {"role": "system", "content": system_prompt},
        ]

        self.debug = debug
        
    def _init_client(self):
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        if self.debug: print(f"Initialized {self._name} client.")

    def chat(self, text: str) -> str:
        res = self.client.messages.create(
            model=self.model_var,
            messages=self.messages + [{"role": "user", "content": text}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self.messages.append({"role": res['role'], "content": res['content']})
        return self.messages[-1]['content']

    def chat_stream(self, text: str) -> Generator[str, None, None]:
        with self.client.messages.stream (
            model=self.model_var,
            messages=self.messages + [{"role": "user", "content": text}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ) as res_stream:
            
            new_message = {"role": "assistant", "content": ""}
            for text in res_stream.text_stream:
                new_message["content"] += text
                yield text
            self.messages.append(new_message)

class CohereChat(ChatABC):
    def __init__(self, model: str):
        super().__init__(model)

    def chat(self, text: str) -> str:
        pass

    def chat_stream(self, text: str) -> Generator[str, None, None]:
        pass

class InstructorOpenAIChat(ChatABC):
    pass

class InstructorAnthropicChat(ChatABC):
    pass

# TODO: Then we'll have a factory function/class that will return the correct chat class based on the model
    
class Chat:
    _models = MODELS

    def __new__(cls, model: str):
        if model not in cls._models: raise ValueError(f"Model {model} not in {cls._models}")
        if model == "openai": return OpenAIChat(model)
        elif model == "anthropic": return AnthropicChat(model)
        elif model == "cohere": return CohereChat(model)
        else: raise ValueError(f"Model {model} not supported")