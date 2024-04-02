from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd

class Query(BaseModel):
    text: str = Field(description="The text to be searched for")

class Queries(BaseModel):
    query1: Query = Field(description="The first query")
    query2: Query = Field(description="The second query")
    query3: Query = Field(description="The third query", default=None)

class Problem(BaseModel):
    problem: str
    problem_details: str
    subproblems: List[str]
    market_profile: str
    customer_profile: str

class Problems(BaseModel):
    problems: List[Problem]

    def parse(self):
        """Parse the object into a list of dictionaries"""
        return [i.__dict__ for i in self.problems]
    
    def __str__(self):
        """String representation of the object"""
        out = ""
        for i, item in enumerate(self.problems):
            out += f"--------Problem {i+1} --------\n"
            for k, v in item.__dict__.items():
                out += f"{k}: {v}\n"
            out += "\n"
        return out
    
    def print(self):
        """Print the string representation of the object"""
        print(self.__str__())

    @property
    def df(self): return pd.DataFrame(self.parse())

class Solution(BaseModel):
    problem: str = Field(description="One of the provided problems")
    solutions_dict: List[Dict[str, str]] = Field(description="A list of dictionaries of solution names and their details")

class Solutions(BaseModel):
    solutions: List[Solution]

    def parse(self):
        """Parse the object into a list of dictionaries"""
        flat_data = []
        for solution in self.solutions:
            problem = solution.problem
            for sd in solution.solutions_dict:
                for solution_name, details in sd.items():
                    flat_data.append({
                        "Problem": problem,
                        "Solution Name": solution_name,
                        "Details": details
                    })
        return flat_data
    
    def __str__(self):
        """String representation of the object"""
        out = ""
        for i, item in enumerate(self.parse()):
            out += f"--------Solution {i+1} --------\n"
            for k, v in item.__dict__.items():
                out += f"{k}: {v}\n"
            out += "\n"
        return out
    
    def print(self):
        """Print the string representation of the object"""
        print(self.__str__())

    @property
    def df(self): return pd.DataFrame(self.parse())