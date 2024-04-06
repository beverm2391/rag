from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from server.config import debug
from lib.utils import load_env

env = load_env(['X-API-KEY'])

class AuthMiddleWare(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        api_key = request.headers.get("X-API-KEY") # Get the api key from the request headers

        if api_key is None or api_key != env["X-API-KEY"]: # Check if the api key is valid
            if debug: print(f"Key provided: {api_key}")
            return JSONResponse(content={"message": "Unauthorized"}, status_code=401)

        return await call_next(request)