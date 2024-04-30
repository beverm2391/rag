from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from time import perf_counter

from models.chat import ChatRequest, ChatResponse

from lib.chat import Chat
from server.config import DEFAULTS, logger, debug_bool
from lib.utils import stream_dicts_as_json
from time import sleep
import json

router = APIRouter()

# ! Load global variables ========================
assert (default_model := DEFAULTS.get('model', None)) is not None, "Default model not set"
assert (default_max_tokens := DEFAULTS.get('max_tokens', None)) is not None, "Default max_tokens not set"
assert (default_temperature := DEFAULTS.get('temperature', None)) is not None, "Default temperature not set"


# ! Routes ========================
@router.get("/")
def endpoint_chat_root():
    return JSONResponse(content={"message": "This is chat()"}, status_code=200)

@router.get("/stream")
def endpoint_chat_stream():
    return JSONResponse(content={"message": "This is chat_stream()"}, status_code=200)

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

    prompt = messages[-1]['content']
    rest = messages[:-1] if len(messages) > 1 else None

    # Create the chat object ========================
    # ! The robust error handling here prevents the server from crashing if an error occurs
    try:
        chat = Chat.create(
            model, temperature, max_tokens,
            system_prompt=system_prompt, messages=rest, debug=debug_bool
        )
    except Exception as e:
        logger.error(f"Error in Chat.create(): {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
    # Get the last message as the prompt
    prompt = messages[-1]['content']
    
    try:        
        return chat.chat(prompt)
    except Exception as e:
        logger.error(f"Error in chat(): {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Streaming chat endpoint ========================
@router.post("/stream")
def endpoint_chat_stream_post(req: ChatRequest) -> StreamingResponse:

    # Parse the request ========================
    messages, model, max_tokens, temperature, system_prompt = _parse_chat_request(req)

    prompt = messages[-1]['content']
    rest = messages[:-1] if len(messages) > 1 else None

    # Create the chat object ========================
    # ! The robust error handling here prevents the server from crashing if an error occurs
    try:
        chat = Chat.create(
            model, temperature, max_tokens,
            system_prompt=system_prompt, messages=rest, debug=debug_bool
        )
    except Exception as e:
        logger.error(f"Error in Chat.create(): {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

    try:
        stream = stream_dicts_as_json(chat.chat_stream(prompt)) # ? Convert the dict generator to a stream of JSON strings
        return StreamingResponse(stream, media_type="text/event-stream") 
    except Exception as e:
        logger.error(f"Error in chat_stream(): {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.post("/test")
def endpoint_chat_test(req: ChatRequest) -> StreamingResponse:
    def example_generator_every_second():
        for i in range(10):
            yield json.dumps({"type": "test", "data": f"Hello, {i}"})
            sleep(1)
        yield json.dumps({"type": "object", "data": "end of stream"})

    return StreamingResponse(example_generator_every_second(), media_type="text/event-stream")