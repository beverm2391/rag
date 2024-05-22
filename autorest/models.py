from typing import Dict, List, Optional
from pydantic import BaseModel, RootModel


class DatabaseModels(RootModel):
    root: Dict[str, str]

class Route(BaseModel):
    route: str
    method: str
    function: str


class RouteGroup(BaseModel):
    routes: List[Route]


class RouteConfig(RootModel):
    root: Dict[str, RouteGroup]


# For Generators
class GeneratedPydanticModel(BaseModel):
    table_name: str
    imports: str
    code: str

class RegeneratedCode(BaseModel):
    code: str
    cause_of_error: Optional[str] = None