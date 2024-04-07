from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
import asyncio
import time

router = APIRouter()

# ! NAMING CONVENTION TO AVOID CONFLICTS
# ? "endpoint" + "router" + "method"

@router.get("/")
def endpoint_example_root():
    return {"message": "Hello World!"}

# Example streaming
@router.get("/streaming-sync")
def endpoint_example_streaming() -> StreamingResponse:
    def example_generator():
        for i in range(10):
            yield f"data: {i}\n\n"
            time.sleep(1)
    return StreamingResponse(example_generator(), media_type="text/event-stream")

# Example async
@router.get("/streaming-async")
async def endpoint_example_streaming_async() -> StreamingResponse:
    async def example_generator():
        for i in range(10):
            yield f"data: {i}\n\n"
            await asyncio.sleep(1)
    return StreamingResponse(example_generator(), media_type="text/event-stream")