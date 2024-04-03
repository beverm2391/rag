from abc import ABC, abstractmethod
from typing import Generator, Dict, List
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

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model_name}, temperature={self.temperature}, max_tokens={self.max_tokens}, system_prompt={self.system_prompt if len(self.system_prompt) < 20 else self.system_prompt[:20] + '...'}, debug={self.debug})"

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
        self.model_var = model_var
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
        response_text = res.choices[0].message.content
        self.messages.append({"role": "assistant", "content": response_text})
        # TODO - Add support for tool caling, other stuff sent back
        return {
            "type" : "object",
            "data": {
                "text": response_text,
            },
        }

    # ? DOCS: https://platform.openai.com/docs/api-reference/chat/streaming
    def chat_stream(self, text: str) -> Generator[Dict[str, any], None, None]:
        res_stream = self.client.chat.completions.create(
            model=self.model_var,
            messages=self.messages + [{"role": "user", "content": text}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        new_message = {"role": "assistant", "content": ""}

        for chunk in res_stream:
            if chunk.choices[0].delta.get("content"):
                content = chunk.choices[0].delta["content"]
                new_message["content"] += content
                yield {"type": "text", "data": content}

            # TODO - Add support for tool caling, other stuff sent back 
            if chunk.choices[0].finish_reason:
                self.messages.append(new_message)
                yield {
                    "type": "object",
                    "data": {
                        "text": new_message["content"],
                    },
                }

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
        self.model_var = model_var
        self.context_window = context_window

        self.messages = [] # claude doesn't use a system message like OAI
        # ? DOCS: https://docs.anthropic.com/claude/docs/system-prompts

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
            system=self.system_prompt,
        )
        # print("res: ", res)
        # print("res.content: ", res.content)
        response_text = res.content[0].text
        self.messages.append({"role": res.role, "content": response_text})
        return {
            "type": "object",
            "data": {
                "text": response_text ,
            }
        }

    # ? DOCS: https://docs.anthropic.com/claude/reference/messages-streaming
    def chat_stream(self, text: str) -> Generator[str, None, None]:
        with self.client.messages.stream (
            model=self.model_var,
            messages=self.messages + [{"role": "user", "content": text}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ) as res_stream:
            
            new_message = {"role": "assistant", "content": ""}
            
            for event in res_stream.text_stream:
                if event.get('type') == "mesage_start":
                    pass
                elif event.get('type') == "content_block_start":
                    text = event['content_block']['text']
                    new_message["content"] += text
                    yield {"type": "text", "data": text} # content_block : {"type": "text", "text": "" }
                elif event.get('type') == "content_block_delta":
                    text = event['delta']['text']
                    new_message["content"] += text
                    yield {"type": "text", "data": text}
                elif event.get('type') == "content_block_stop":
                    pass
                elif event.get('type') == "message_delta":
                    # data: {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence":null, "usage":{"output_tokens": 15}}}
                    pass
                elif event.get('type') == "message_stop":
                    yield {
                        "type": "object",
                        "data": {
                            "text": new_message["content"],
                        },
                    }

            self.messages.append(new_message)

class CohereChat(ChatABC):
    """Chat with Cohere"""
    _name = "cohere"
    _api_key = COHERE_API_KEY

    def __init__(
            self, model_name: str, temperature: float, max_tokens: int, system_prompt: str, debug: bool = False,
            model_var: str = None, context_window: int = 0, web_search: bool = False, citations: bool = False):
        super().__init__(model_name)
        self._init_client()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.model_var = model_var
        self.context_window = context_window
        self.web_search = web_search
        self.citations = citations

        self.messages = [
            {"role": "SYSTEM", "message": system_prompt},
        ]

        self.debug = debug
        
    def _init_client(self):
        self.client = cohere.Client(api_key=COHERE_API_KEY)
        if self.debug: print(f"Initialized {self._name} client.")

    def chat(self, text: str) -> str:
        res = self.client.chat(
            model=self.model_var,
            chat_history=self.messages,
            message=text, # ? DOCS: https://docs.cohere.com/reference/chat
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            connectors=[{"id": "web-search"}] if self.web_search else None,
        )

        self.messages.append({"role": "USER", "message": text})
        self.messages.append({"role": "CHATBOT", "message": res.text})

        return {
            "type": "object",
            "data": {
                "text": res.text,
                "citations": res.citations,
                "search_results": res.search_results,
                "documents": res.documents,
                "search_queries": res.search_queries,
            },
        }

    # ? DOCS: https://docs.cohere.com/docs/streaming#retrieval-augmented-generation-stream-events
    def chat_stream(self, text: str) -> Generator[str, None, None]:
        res_stream = self.client.chat(
            model=self.model_var,
            chat_history=self.messages,
            message={"role": "USER", "message": text},
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            connectors=[{"id": "web-search"}] if self.web_search else None,
            stream=True,
        )

        new_message = {"role": "CHATBOT", "message": ""}

        # handle each event
        for event in res_stream:
            if event['event_type'] == "stream_start":
                pass
            elif event['event_type'] == "text-generation":
                new_message["message"] += event['text']
                yield {"type": "text", "data": event.text}
            elif event['event_type'] == "search-results":                
                # search_results.extend(event['search_results'])
                # documents.extend(event['documents'])
                pass
            elif event['event_type'] == "citation-generation":
                # citations.extend(event['citations'])
                pass
            elif event['event_type'] == "stream_end":
                self.messages.append(new_message)
                yield {
                    "type": "object",
                    "data": {
                        "text": new_message["message"], 
                        "token_count": event['response']['token_count'],
                        "citations": event['response'].get("citations", []),
                        "search_results": event['response'].get("search_results", []),
                        "documents": event['response'].get("documents", []),
                        "search_queries": event['response'].get("search_queries", []),
                    },
                }

class InstructorOpenAIChat(ChatABC):
    pass

class InstructorAnthropicChat(ChatABC):
    pass

# TODO: Then we'll have a factory function/class that will return the correct chat class based on the model
    
class Chat:
    _models = MODELS

    def __new__(cls,
            model_name: str, temperature: float, max_tokens: int, system_prompt: str,
            debug: bool = False, **kwargs
            ):

        assert model_name in cls._models, f"Model {model_name} not found in model config"

        _model = cls._models[model_name]
        model_var = _model.get('model_var', None)
        org = _model.get('org', None)
        context_window = _model.get('context_window', None)

        assert org is not None, f"Organization missing for {model_name} in model config"
        assert model_var is not None, f"Model Var missing for {model_name} in model config"
        assert context_window is not None, f"Context Window missing for {model_name} in model config"

        if debug:
            print(f"Model_name: {model_name}")
            print(f"Model Var: {model_var}")
            print(f"Org: {org}")
            print(f"Context Window: {context_window}")

        if org == "openai":
            if debug: print(f"Creating {model_name} chat instance.")
            return OpenAIChat(model_name, temperature, max_tokens, system_prompt, model_var=model_var, context_window=context_window, debug=debug, **kwargs)
        elif org == "anthropic":
            if debug: print(f"Creating {model_name} chat instance.")
            return AnthropicChat(model_name, temperature, max_tokens, system_prompt, model_var=model_var, context_window=context_window, debug=debug, **kwargs)
        elif org == "cohere":
            if debug: print(f"Creating {model_name} chat instance.")
            return CohereChat(model_name, temperature, max_tokens, system_prompt, model_var=model_var, context_window=context_window, debug=debug, **kwargs)
        else: raise ValueError(f"Model {model_name} not supported")