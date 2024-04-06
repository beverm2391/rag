from fastapi import APIRouter, Depends

from models.chat import ChatRequest, ChatResponse

from lib.chat import Chat
from server.config import DEFAULTS, debug

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

@router.post("/")
def endpoint_chat_root_post(req: ChatRequest) -> ChatResponse:
    

    # ! Parse the request ========================
    messages = [m.model_dump(exclude_none=True) for m in req.messages] # ? Convert the pydantic model to a dictionary
    model = req.model if req.model else default_model
    max_tokens = req.max_tokens if req.max_tokens else default_max_tokens
    temperature = req.temperature if req.temperature else default_temperature
    system_prompt = DEFAULTS['system_prompt']

    if debug:
        print("Args passed to chat():")
        print(f"model: {model}")
        print(f"max_tokens: {max_tokens}")
        print(f"temperature: {temperature}")
        print(f"system_prompt: {system_prompt}")

    # ! Create the chat object ========================
    chat = Chat.create(model, temperature, max_tokens, system_prompt, debug=debug)

    if debug:
        print("Chat object created")
        print(f"text to be passed: {messages}")

    prompt = messages[-1]['content']
    return chat.chat(prompt)