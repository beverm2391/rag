from fastapi import APIRouter
from fastapi.responses import JSONResponse

from lib.model_config import MODELS, DEFAULTS

router = APIRouter()

@router.get("/")
def endpoint_get_model_config():  
    data = {
        'models': MODELS,
        'defaults': DEFAULTS
    } 
    return JSONResponse(content=data, status_code=200)

@router.get("/{model_name}")
def endpoint_get_model_config_by_name(model_name: str):
    if model_name in MODELS: return JSONResponse(content=MODELS[model_name], status_code=200)
    else: return JSONResponse(content={"message": "Model not found"}, status_code=404)

@router.get("/defaults")
def endpoint_get_default_config():
    return JSONResponse(content=DEFAULTS, status_code=200)