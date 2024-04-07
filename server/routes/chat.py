from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from time import perf_counter

from models.chat import ChatRequest, ChatResponse

from lib.chat import Chat
from server.config import DEFAULTS, logger, debug_bool
from lib.utils import stream_dicts_as_json, stream_dicts_as_json_async

router = APIRouter()

# ! Load global variables ========================
assert (default_model := DEFAULTS.get('model', None)) is not None, "Default model not set"
assert (default_max_tokens := DEFAULTS.get('max_tokens', None)) is not None, "Default max_tokens not set"
assert (default_temperature := DEFAULTS.get('temperature', None)) is not None, "Default temperature not set"


# ! Routes ========================
@router.get("/")
def endpoint_chat_root():
    return {"message": "This is chat()"}


@router.get("/stream")
def endpoint_chat_stream():
    return {"message": "This is chat_stream()"}

def _parse_chat_request(req: ChatRequest) -> tuple:
    # ! Parse the request ========================
    messages = [m.model_dump(exclude_none=True) for m in req.messages] # ? Convert the pydantic model to a dictionary
    model = req.model if req.model else default_model
    max_tokens = req.max_tokens if req.max_tokens else default_max_tokens
    temperature = req.temperature if req.temperature else default_temperature
    system_prompt = DEFAULTS['system_prompt']

    logger.debug("Args passed to chat():")
    logger.debug(f"model: {model}")
    logger.debug(f"max_tokens: {max_tokens}")
    logger.debug(f"temperature: {temperature}")
    logger.debug(f"system_prompt: {system_prompt}")

    return messages, model, max_tokens, temperature, system_prompt

# Regular chat endpoint ========================
@router.post("/")
def endpoint_chat_root_post(req: ChatRequest) -> ChatResponse:
    
    # Parse the request ========================
    messages, model, max_tokens, temperature, system_prompt = _parse_chat_request(req)

    # Create the chat object ========================
    chat = Chat.create(model, temperature, max_tokens, system_prompt, debug=debug_bool)
    
    # Get the last message as the prompt
    prompt = messages[-1]['content']

    return chat.chat(prompt)


# Streaming chat endpoint ========================
@router.post("/stream")
def endpoint_chat_stream_post(req: ChatRequest) -> StreamingResponse:

    # Parse the request ========================
    messages, model, max_tokens, temperature, system_prompt = _parse_chat_request(req)

    # Create the chat object ========================
    chat = Chat.create(model, temperature, max_tokens, system_prompt, debug=debug_bool)

    # Get the last message as the prompt, and create a stream 
    prompt = messages[-1]['content']
    stream = stream_dicts_as_json(chat.chat_stream(prompt)) # ? Convert the dict generator to a stream of JSON strings
    return StreamingResponse(stream, media_type="text/event-stream") 