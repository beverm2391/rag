from abc import ABC, abstractmethod
from typing import Generator, Dict, List, Type, Union, Literal, Callable, Optional
from pydantic import BaseModel
from openai import OpenAI
from anthropic import Anthropic
import cohere
import json
from time import perf_counter

from lib.utils import load_env
from lib.model_config import MODELS

# TODO: The plan here is to have support for OpenAI, Anthropic, Cohere, and so on in one unified API
# This is so we can just plug the same code into any of these services and have it work
# Ideally we'll have a rag class that 

# chat class takes text in, and returns text out via a chat method
# expose the same chat methods (regular, streaming, etc) for all services
# Then a rag class can just take in an Index and a Chat class and work with any service

# lets make an abstract class for chat

env = load_env()
OPENAI_API_KEY = env.get("OPENAI_API_KEY", None)
ANTHROPIC_API_KEY = env.get("ANTHROPIC_API_KEY", None)
COHERE_API_KEY = env.get("COHERE_API_KEY", None)


# ! Chat Abstract Class ===============================

class ChatABC(ABC):
    """Abstract base class for chat services."""
    def __init__(
            self, model_name: str, temperature: float = 0, max_tokens: int = 2048, system_prompt: str = "You are a helpful assistant.",
            debug: bool = False, as_json: bool = False, persist=False, **kwargs
        ):
        self.model_name = model_name
        self.model_var = None
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.debug = debug
        self.as_json = as_json
        self.persist = persist

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

    def print_stream(self, text: str, line_size: int = 22) -> Dict[str, any]:
        """Print the response text as it comes in, and return the final response object."""
        res_list = []

        for i, response in enumerate(output:=self.chat_stream(text)):
            if response is None:
                continue # skip empty responses
            if response["type"] == "text": # if response is text, print it
                text = response["data"]
                if i % line_size == 0 and i !=0:
                    print(text) # print response with new line every 22 responses
                else:
                    print(text, end="") # print response without new line
            elif response["type"] == "object": # if response is an object, append it to a list to be returned
                res_list.append(response)

        return res_list # return the list of response objects

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model_name}, temperature={self.temperature}, max_tokens={self.max_tokens}, system_prompt={self.system_prompt if len(self.system_prompt) < 20 else self.system_prompt[:20] + '...'}, debug={self.debug})"


# ! Service Specific Chat Implementations ===============================

class OpenAIChat(ChatABC):
    """Chat with OpenAI"""
    _name = "openai"
    _api_key = OPENAI_API_KEY

    def __init__(
            self, model_name: str, temperature: float, max_tokens: int, system_prompt: str, messages: List[Dict],
            debug: bool = False, model_var: str = None, context_window: int = 0, as_json: bool = False, persist=False
        ):
        super().__init__(model_name)
        self._init_client()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.model_var = model_var
        self.context_window = context_window
        self.as_json = as_json
        self.persist = messages or persist # we always persist if messages are passed
        self.debug = debug

        # if no messages are passed, reset messages list with system prompt
        if not messages: self.messages = [{"role": "system", "content": system_prompt}]
        
    def _init_client(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        if self.debug: print(f"Initialized {self._name} client.")

    def chat(self, text: str) -> str:
        if not self.persist: self.messages = [{"role": "system", "content": self.system_prompt}]  # reset messages list if persist is False
        self.messages.append({"role": "user", "content": text}) # add user message to messages list
        res = self.client.chat.completions.create(
            model=self.model_var,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        response_text = res.choices[0].message.content
        self.messages.append({"role": "assistant", "content": response_text})  # add assistant message to messages list
        # TODO - Add support for tool caling, other stuff sent back
        return {
            "type" : "object",
            "data": {
                "text": response_text,
            },
        }

    # ? DOCS: https://platform.openai.com/docs/api-reference/chat/streaming
    def chat_stream(self, text: str) -> Generator[Dict[str, any], None, None]:
        if not self.persist: self.messages = [{"role": "system", "content": self.system_prompt}]  # reset messages list if persist is False
        self.messages.append({"role": "user", "content": text}) # add user message to messages list
        res_stream = self.client.chat.completions.create(
            model=self.model_var,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        new_message = {"role": "assistant", "content": ""}

        for chunk in res_stream:
            # if self.debug: print(f"event came in from external API at {perf_counter() * 1000:0.2f}ms")
            if chunk.choices[0].delta:
                if content := chunk.choices[0].delta.content:
                    new_message["content"] += content
                    dict_ = {"type": "text", "data": content}
                    yield json.dumps(dict_) if self.as_json else dict_

            # TODO - Add support for tool caling, other stuff sent back 
            if chunk.choices[0].finish_reason:
                self.messages.append(new_message)
                dict_ = {
                    "type": "object",
                    "data": {
                        "text": new_message["content"],
                    },
                }
                yield json.dumps(dict_) if self.as_json else dict_

        # yield "event: end\ndata: \n\n"  # Signal the end of the stream
        # TODO delete this??


class AnthropicChat(ChatABC):
    """Chat with Anthropic"""
    _name = "anthropic"
    _api_key = ANTHROPIC_API_KEY

    def __init__(
            self, model_name: str, temperature: float, max_tokens: int, system_prompt: str, messages: List[Dict],
            debug: bool = False, model_var: str = None, context_window: int = 0, as_json: bool = False, persist=False
        ):
        super().__init__(model_name)
        self._init_client()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.model_var = model_var
        self.context_window = context_window
        self.as_json = as_json
        self.persist = persist

        self.messages = [] # claude doesn't use a system message like OAI
        # ? DOCS: https://docs.anthropic.com/claude/docs/system-prompts

        self.debug = debug
        
    def _init_client(self):
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        if self.debug: print(f"Initialized {self._name} client.")

    def chat(self, text: str) -> str:
        if not self.persist: self.messages = []  # reset messages list if persist is False
        self.messages.append({"role": "user", "content": text})
        res = self.client.messages.create(
            model=self.model_var,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
        )
        # print("res: ", res)
        # print("res.content: ", res.content)
        response_text = res.content[0].text
        self.messages.append({"role": res.role, "content": response_text})
        if not self.persist: self.messages = []  # reset messages list if persist is False
        return {
            "type": "object",
            "data": {
                "text": response_text ,
            }
        }

    # ? DOCS: https://docs.anthropic.com/claude/reference/messages-streaming
    def chat_stream(self, text: str) -> Generator[str, None, None]:
        if not self.persist: self.messages = []  # reset messages list if persist is False
        self.messages.append({"role": "user", "content": text})
        with self.client.messages.stream (
            model=self.model_var,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ) as res_stream:
            
            new_message = {"role": "assistant", "content": ""}
            
            for event in res_stream:
                # if self.debug: print(f"event came in from external API at {perf_counter() * 1000:0.2f}ms")
                if event.type == "mesage_start":
                    pass
                elif event.type == "content_block_start":
                    text = event.content_block.text
                    new_message["content"] += text
                    dict_ = {"type": "text", "data": text} # content_block : {"type": "text", "text": "" }
                    yield json.dumps(dict_) if self.as_json else dict_
                elif event.type == "content_block_delta":
                    text = event.delta.text
                    new_message["content"] += text
                    dict_ = {"type": "text", "data": text}
                    yield json.dumps(dict_) if self.as_json else dict_
                elif event.type == "content_block_stop":
                    pass
                elif event.type == "message_delta":
                    # data: {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence":null, "usage":{"output_tokens": 15}}}
                    pass
                elif event.type == "message_stop":
                    dict_ = {
                        "type": "object",
                        "data": {
                            "text": new_message["content"],
                        },
                    }
                    yield json.dumps(dict_) if self.as_json else dict_
                    self.messages.append(new_message)


class CohereChat(ChatABC):
    """Chat with Cohere"""
    _name = "cohere"
    _api_key = COHERE_API_KEY

    def __init__(
            self, model_name: str, temperature: float, max_tokens: int, system_prompt: str, messages: List[Dict],
            debug: bool = False, model_var: str = None, context_window: int = 0, web_search: bool = False,
            citations: bool = False, as_json: bool = False, persist=False
        ):
        super().__init__(model_name)
        self._init_client()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.model_var = model_var
        self.context_window = context_window
        self.web_search = web_search
        self.citations = citations
        self.as_json = as_json
        self.persist = persist

        self.messages = [
            {"role": "SYSTEM", "message": system_prompt},
        ]

        self.debug = debug
        
    def _init_client(self):
        self.client = cohere.Client(api_key=COHERE_API_KEY)
        if self.debug: print(f"Initialized {self._name} client.")

    def chat(self, text: str) -> str:
        if not self.persist: self.messages = [{"role": "SYSTEM", "message": self.system_prompt}]
        res = self.client.chat(
            model=self.model_var,
            chat_history=self.messages,
            message=text, # ? DOCS: https://docs.cohere.com/reference/chat
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            connectors=[{"id": "web-search"}] if self.web_search else None,
        )

        self.messages.append({"role": "USER", "message": text}) # append after because cohere takes separate chat history and query
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
        if not self.persist: self.messages = [{"role": "SYSTEM", "message": self.system_prompt}]
        res_stream = self.client.chat_stream(
            model=self.model_var,
            chat_history=self.messages,
            message=text,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            connectors=[{"id": "web-search"}] if self.web_search else None,
        )

        self.messages.append({"role": "USER", "message": text})
        new_message = {"role": "CHATBOT", "message": ""}

        for event in res_stream:
            # if self.debug: print(f"event came in from external API at {perf_counter() * 1000:0.2f}ms")
            if event.event_type == "stream_start":
                pass
            elif event.event_type == "search-queries-generation":
                dict_ = {"type": "object", "data": event.search_queries}
                yield json.dumps(dict_) if self.as_json else dict_
            elif event.event_type == "text-generation":
                new_message["message"] += event.text
                dict_ = {"type": "text", "data": event.text}
                yield json.dumps(dict_) if self.as_json else dict_
            elif event.event_type == "search-results":                
                dict_ = {"type": "object", "data": {
                    "search_results": event.search_results,
                    "documents": event.documents,
                }}
                yield json.dumps(dict_) if self.as_json else dict_
            elif event.event_type == "citation-generation":
                dict_ = {"type": "object", "data": event.citations}
                yield json.dumps(dict_) if self.as_json else dict_
            elif event.event_type == "stream-end":
                self.messages.append(new_message)
                dict_ = {
                    "type": "object",
                    "data": {
                        "text": new_message["message"], 
                        "tool_calls": event.response.tool_calls,
                        "citations": event.response.citations,
                        "search_results": event.response.search_results,
                        "documents": event.response.documents,
                        "search_queries": event.response.search_queries,
                        "is_search_required": event.response.is_search_required,
                        "meta" : event.response.meta,
                        "chat_history": [{"role" : m['role'], "message": m['message']} for m in self.messages],
                    },
                }
                yield json.dumps(dict_) if self.as_json else dict_

class InstructorOpenAIChat(ChatABC):
    pass

class InstructorAnthropicChat(ChatABC):
    pass


# ! Message Converter Schemas ========================
# ? These models are used to validate the messages before and after conversion 

class UniveralMessage(BaseModel):
    role: Literal['system', 'user', 'assistant']
    content: str

class OpenAIMessage(BaseModel):
    role: Literal['system', 'user', 'assistant']
    content: str

class AnthropicMessage(BaseModel):
    role: Literal['system', 'user', 'assistant'] # ! even though Anthropic doesnt use a system message, I am including it here for consistency as it will be handled much later, right before the API call
    content: str

class CohereMessage(BaseModel):
    role: Literal['SYSTEM', 'USER', 'CHATBOT'] 
    message: str

class UniversalMessages(BaseModel):
    messages: List[UniveralMessage]\
    
    def to_messages(self): return [message.model_dump() for message in self.messages]

class OpenAIMessages(BaseModel):
    messages: List[OpenAIMessage]

    def to_messages(self): return [message.model_dump() for message in self.messages]

class AnthropicMessages(BaseModel):
    messages: List[AnthropicMessage]

    def to_messages(self): return [message.model_dump() for message in self.messages]

class CohereMessages(BaseModel):
    messages: List[CohereMessage]

    def to_messages(self): return [message.model_dump() for message in self.messages]

# The typehint for all possible message types
MessageTypes = Union[OpenAIMessages, AnthropicMessages, CohereMessages]

# ! Chat Factory/Static Class ========================
# ? This class is used to create chat instances and convert messages between different formats

class Chat:
    _models = MODELS

    @staticmethod
    def get_model_config(): return MODELS

    @classmethod
    def create(
        cls,
        model_name: str, temperature: float, max_tokens: int, system_prompt: str,
        messages: List[Dict] = None, debug: bool = False, as_json: bool = False, persist: bool = False, **kwargs
        ) -> Union['OpenAIChat', 'AnthropicChat', 'CohereChat']:

        assert model_name in cls._models, f"Model {model_name} not found in model config"
        model = cls._models[model_name]
        model_var = model.get('model_var', None)
        org = model.get('org', None)
        context_window = model.get('context_window', None)

        assert org is not None, f"Organization missing for {model_name} in model config"
        assert model_var is not None, f"Model Var missing for {model_name} in model config"
        assert context_window is not None, f"Context Window missing for {model_name} in model config"
        assert context_window >= max_tokens, f"Context window {context_window} must be greater than max tokens {max_tokens}"

        if org == "openai":
            if debug: print(f"Creating {model_name} chat instance.")
            instance: OpenAIChat = OpenAIChat(
                model_name, temperature, max_tokens, system_prompt, messages,
                model_var=model_var, context_window=context_window, debug=debug, as_json=as_json, persist=persist, **kwargs
            )
        elif org == "anthropic":
            if debug: print(f"Creating {model_name} chat instance.")
            instance: AnthropicChat = AnthropicChat(
                model_name, temperature, max_tokens, system_prompt, messages,
                model_var=model_var, context_window=context_window, debug=debug, as_json=as_json, persist=persist, **kwargs
            )
        elif org == "cohere":
            if debug: print(f"Creating {model_name} chat instance.")
            instance: CohereChat = CohereChat(
                model_name, temperature, max_tokens, system_prompt, messages,
                model_var=model_var, context_window=context_window, debug=debug, as_json=as_json, persist=persist, **kwargs
            )
        else:
            raise ValueError(f"Model {model_name} not supported")

        return instance
    
    # ? Message Conversion ========================

    def universal_validator(func: Callable) -> Callable:
        def wrapper(messages: UniversalMessages) -> MessageTypes:
            try:
                validated = UniversalMessages(messages=[UniveralMessage(**message) for message in messages])
            except Exception as e:
                print(e)
                raise ValueError(f'Messages passed to {func.__name__} are not in the correct format, as defined by the `UniversalMessages` model')
            return func(messages)
        return wrapper


    @staticmethod
    @universal_validator
    def convert_messages_to_openai(messages: UniversalMessages) -> OpenAIMessages:
        try:
            return OpenAIMessages(messages=[OpenAIMessage(**message) for message in messages]).to_messages()
        except Exception as e:
            print(e)
            raise ValueError('Failed to convert messages to OpenAI format')

    @staticmethod
    @universal_validator
    def convert_messages_to_anthropic(messages: UniversalMessages) -> AnthropicMessages:
        # ! Anthropic does not use a system message, but we leave that in for now for compatibility/simplicity
        # ! We will remove the system message later (at the very last point within the funciton that calls the API)
        try:
            return AnthropicMessages(messages=[AnthropicMessage(**message) for message in messages]).to_messages()
        except Exception as e:
            print(e)
            raise ValueError('Failed to convert messages to Anthropic format')

    @staticmethod
    @universal_validator
    def convert_messages_to_cohere(messages: UniversalMessages) -> CohereMessages:
        new_messages = []
        for message in messages:

            # convert the role to the correct format
            if message['role'] == 'system': role = 'SYSTEM'
            elif message['role'] == 'user': role = 'USER'
            elif message['role'] == 'assistant': role = 'CHATBOT'
            else: raise ValueError(f'Unsupported input role: {message["role"]}')

            # create a new message with keys 'role' and 'message' (instead of 'content')
            new_messages.append({'role': role, 'message': message['content']})
        try:
            return CohereMessages(messages=[CohereMessage(**message) for message in new_messages]).to_messages()
        except Exception as e:
            print(e)
            raise ValueError('Failed to convert messages to Cohere format')