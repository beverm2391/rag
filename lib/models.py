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
        for key, value in self.__dict__.items():
            if key != "_raw_response" and value: data.append(value.parse())
        return data

    def get_query_text_only(self):
        """Get the text of the queries"""
        return [query["text"] for query in self.parse()]