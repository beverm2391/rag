from __future__ import annotations
import openai
import instructor
from typing import List, Union, Optional
from time import perf_counter
import asyncio

from lib.utils import load_env
from lib.models import Queries, Query

vars = load_env(['OPENAI_API_KEY']) # load all vars, ensuring that a env exists and the expected vars are present
OPENAI_API_KEY = vars['OPENAI_API_KEY']

class GeneratorSync:
    def __init__(self, output_schema, system_prompt: str = "You are a helpful assistant.", model: str = "gpt-3.5-turbo", debug: bool = False):
        self.client = instructor.patch(openai.OpenAI(api_key=OPENAI_API_KEY))
        self.output_schema = output_schema
        self.model = model
        self.messages = [{"role": "system", "content": system_prompt}]
        self.debug = debug

    def generate(self, query: str, model: str = None):
        self.messages.append({"role": "user", "content": query})

        if self.debug:
            t = perf_counter()
            print("System Prompt:", self.messages[0]['content'])
            print("User Prompt:", self.messages[1]['content'])
            print("Model:", model or self.model)
            print("Output Schema:", self.output_schema)
            print("Generating...")

        res = self.client.chat.completions.create(
            model=model or self.model,
            response_model=self.output_schema,
            messages = self.messages,
        )

        if self.debug: print(f"Response in: {perf_counter() - t:.2f}s")
        return res
    

class GeneratorAsync:
    _sem = None # class var

    def __init__(self, output_schema, system_prompt: str = "You are a helpful assistant.", api_key: str = None, base_url: str = None, model: str = "gpt-3.5-turbo", rps: int = 10, debug: bool = False):
        self.client = instructor.patch(openai.AsyncOpenAI(api_key=api_key or OPENAI_API_KEY, base_url=base_url))
        self.output_schema = output_schema
        self.model = model
        self.system_prompt = system_prompt
        self.debug = debug
        self.to_complete = []
        
        # handle_ipykernel() # auto nested event loop support from notebook
        if GeneratorAsync._sem is None: GeneratorAsync._sem = asyncio.Semaphore(rps) # init only once (class var)
    
    async def _single_generate(self, query: str, task_id: int = 0, model: str = None):
        start = perf_counter()
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": query}]

        try:
            async with GeneratorAsync._sem:
                res = await self.client.chat.completions.create(
                    model=model or self.model,
                    response_model=self.output_schema,
                    messages = messages,
                )
            if self.debug: print(f"Task id: {task_id} in: {perf_counter() - start:.2f}s")
            self.to_complete[task_id] = True # hash map
            if self.debug: print(f"{sum(1 for i in self.to_complete if i)} of {len(self.to_complete)} tasks completed")

            return res
        except Exception as e:
            if self.debug: print(f"Task Error: {query if len(query) < 20 else query[:20]}... Error: {e}")
            return e # Return error instead of raising it
        
    async def generate(self, queries: list, model: str = None):
        if type(queries) != list: raise ValueError(f"queries must be a list, not {type(queries)}")
        
        start = perf_counter() # start timer
    
        if self.debug: print(f"Generating {len(queries)} queries...")
        self.to_complete = [False] * len(queries) # boolean array/hash map

        tasks = [self._single_generate(query, task_id=i, model=model) for i, query in enumerate(queries)]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        if self.debug:
            print(f"Total time: {perf_counter() - start:.2f}s")
            print(f"Error count: {sum(1 for result in results if isinstance(result, Exception))}")
        return results


class QueryGeneratorAsync(GeneratorAsync):
    def __init__(self, model: str = None, debug: bool = False):
        super().__init__(Queries, system_prompt="You are designed to take in user instructons and generate 2-3 queries to search for data relevant to the user's request.", model=model, debug=debug)

    async def generate(self, user_query_or_queries: Union[List[str], str], model: str = None):
        def _make_prompt(user_insturctions: str):
            intructions = f"""
            INSTRUCITONS:
            Generate 2-3 verbose search queries based on the following user instructions. You must generate a minimum of 2 queries and a maximum of 3 queries. Each query should be unique and relevant to the user's request.        
            """
            example = """
            EXAMPLE 1
            INSTRUCTIONS: What is the impact of deforestation on climate change?
            EXAMPLE RESPONSE:
            1. Query: "relationshipe between deforestation and climate change"
            2. Query: "how are deforestation and climate change related"
            3. Query: "deforestation research global warming effects"

            EXAMPLE 2
            INSTRUCTION: Who is Jim Simons?
            EXAMPLE RESPONSE:
            1. Query: "Jim Simons biography"
            2. Query: "Jim Simons professional background"
            3. Query: "Jim Simons social media"
            """
            query = f"{intructions}\n{example}\nUSER INSTRUCTIONS:\n{user_insturctions}\nOUTPUT:"
            return query
        
        def _handle_input(user_query_or_queries):
            """A function to handle the user input and return the list of prompts to be passed to the generator"""
            if type(user_query_or_queries) == str: return [_make_prompt(user_query_or_queries)] # if user passes a string (one query) (we must still reuturn a list even though its one query)
            elif type(user_query_or_queries == list): return [_make_prompt(query) for query in user_query_or_queries] # if user passes a list of queries
            else: raise ValueError(f"Expected str or list, got {type(user_query_or_queries)}")
        
        queries = _handle_input(user_query_or_queries)
        return await super().generate(queries, model)