from fastapi import APIRouter, Depends

router = APIRouter()

# ! NAMING CONVENTION TO AVOID CONFLICTS
# ? "endpoint" + "router" + "method"

@router.get("/")
def endpoint_example_root():
    return {"message": "Hello World!"}