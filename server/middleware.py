from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from lib.utils import load_env

env = load_env(['SERVER_API_KEY'])

class AuthMiddleWare(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        api_key = request.headers.get("x-api-key") # Get the api key from the request headers

        if api_key is None or api_key != env["SERVER_API_KEY"]: # Check if the api key is valid
            return JSONResponse(content={"message": "Unauthorized"}, status_code=401)

        return await call_next(request)