from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd

class Query(BaseModel):
    text: str = Field(description="The text to be searched for")

    def parse(self): return self.__dict__

class Queries(BaseModel):
    query1: Query = Field(description="The first query")
    query2: Query = Field(description="The second query")
    query3: Optional[Query] = Field(description="The third query", default=None)

    def parse(self):
        """Parse the object into a list of dictionaries"""
        data = []
        for key in self.__dict__.keys(): 
            if key != "_raw_response" and self.__dict__[key]: data.append(self.__dict__[key].parse())
        return data