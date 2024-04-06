from fastapi import FastAPI
from fastapi import Depends
from fastapi.middleware.cors import CORSMiddleware

from middleware import AuthMiddleWare
from server.config import debug

# ! Import the routes ========================
from routes import example, chat

# ! Create the FastAPI app ========================
app = FastAPI()

# ! Add the middleware ========================

app.add_middleware(AuthMiddleWare) # this has to be added before the CORS middleware as to not block the CORS preflight request
if debug: print("Auth middleware added")

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
if debug: print("CORS middleware added")

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