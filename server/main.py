from fastapi import FastAPI
from fastapi import Depends
from fastapi.middleware.cors import CORSMiddleware

from middleware import AuthMiddleWare
from server.config import logger

# ! Set up logging ========================

#? Options:
# CRITICAL = 50
# FATAL = CRITICAL
# ERROR = 40
# WARNING = 30
# WARN = WARNING
# INFO = 20
# DEBUG = 10
# NOTSET = 0

# ! Import the routes ========================
from routes import example, chat

# ! Create the FastAPI app ========================
app = FastAPI()

# ! Add the middleware ========================

app.add_middleware(AuthMiddleWare) # this has to be added before the CORS middleware as to not block the CORS preflight request
logger.debug("Auth middleware added")

# TODO - seal this up
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.debug("CORS middleware added")

# ! Add the routes ========================

# ? https://stackoverflow.com/questions/59965872/how-to-solve-no-attribute-routes-in-fastapi
app.include_router(example.router, prefix="/example", tags=["example"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])

# ! START ROUTES =============================

# ? NAMING CONVENTION TO AVOID CONFLICTS
#  "endpoint" + "router" + "method"

@app.get("/")
async def endpoint_root():
    return {"message": "Hello World!"}

@app.get("/test")
async def endpoint_test():
    return {"message": "Test successful!"}