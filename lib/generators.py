from __future__ import annotations
import openai
import instructor
from typing import List, Union, Optional
from time import perf_counter
import asyncio

from lib.utils import load_env
from lib.models import Problems, Solutions, Queries, Query

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
            Generate 2-3 search queries based on the following user instructions:        
            """
            example = """
            EXAMPLE USER INSTRUCTIONS: I need to find information on the impact of deforestation on climate change.
            EXAMPLE RESPONSE:
            1. Query: "Impact of deforestation on climate change"
            2. Query: "Solutions to deforestation"
            3. Query: "Deforestation statistics by region"
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
    
class ProblemGeneratorSync(GeneratorSync):
    def __init__(self, model: str = None, debug: bool = False):
        super().__init__(Problems, system_prompt="You are designed to assist venture capitlaists in market research. Be concise and to the point, but include all relevant information.", model=model, debug=debug)

    def generate(self, subfield_or_tech: str, model: str = None, n: int = 7):
        instructions = f"""
        INSRUCTIONS:
            1. list {n} problems that {subfield_or_tech} aims to solve
            2. for each problem, include:
                1. details abolut the problem
                2. a list of subproblems
                3. a brief market description for an investor looking to invest in a company that solves this problem
                4. a brief customer profile/description for a customer that would by a solution to this problem
        """
        example = """
        EXAMPLE RESPONSE:
        1. Problem: Deforestation
            - Details: Deforestation contributes to climate change and loss of biodiversity. Caused by activities such as logging, agriculture, and urbanization.
            - Subproblems:
                - Its difficult to monitor tree health
                - Monitoring pests and diseases
                - Managing forest fires
                - Monitoring illegal logging
            - Market Description: The precision forestry market offers advanced technological solutions to combat deforestation effects, such as climate change and biodiversity loss, driven by unsustainable logging, agriculture, and urbanization. It encompasses innovative applications like satellite imagery, drones, AI, and IoT for tree health monitoring, pest and disease control, forest fire management, and illegal logging detection. This rapidly growing sector appeals to investors due to its significant growth potential, alignment with global sustainability goals, and capacity to address critical environmental challenges with cutting-edge solutions.
            - Customer Profile: Forest Management Organizations and Conservation Agencies: These entities manage vast forest areas and seek efficient, scalable tech solutions for sustainable forest management, including health monitoring, pest and disease management, and illegal logging prevention. They face challenges like resource limitations and the need for accurate, timely data. These organizations prioritize investments in technology that enhances operational efficiency, supports sustainability, and provides environmental conservation value.
        ...
        """
        query = f"{instructions}\n{example}"
        return super().generate(query, model)


class ProblemGeneratorAsync(GeneratorAsync):
    def __init__(self, model: str = None, debug: bool = False):
        super().__init__(Problems, system_prompt="You are designed to assist venture capitlaists in market research. Be concise and to the point, but include all relevant information.", model=model, debug=debug)
    
    async def generate(self, queries: List[str], model: str = None, n: int = 7):
        def _make_prompt(query: str):
            instructions = f"""
            INSRUCTIONS:
                1. list {n} problems that {query} aims to solve
                2. for each problem, include:
                    1. details abolut the problem
                    2. a list of subproblems
                    3. a brief market description for an investor looking to invest in a company that solves this problem
                    4. a brief customer profile/description for a customer that would by a solution to this problem
            """
            example = """
            EXAMPLE RESPONSE:
            1. Problem: Deforestation
                - Details: Deforestation contributes to climate change and loss of biodiversity. Caused by activities such as logging, agriculture, and urbanization.
                - Subproblems:
                    - Its difficult to monitor tree health
                    - Monitoring pests and diseases
                    - Managing forest fires
                    - Monitoring illegal logging
                - Market Description: The precision forestry market offers advanced technological solutions to combat deforestation effects, such as climate change and biodiversity loss, driven by unsustainable logging, agriculture, and urbanization. It encompasses innovative applications like satellite imagery, drones, AI, and IoT for tree health monitoring, pest and disease control, forest fire management, and illegal logging detection. This rapidly growing sector appeals to investors due to its significant growth potential, alignment with global sustainability goals, and capacity to address critical environmental challenges with cutting-edge solutions.
                - Customer Profile: Forest Management Organizations and Conservation Agencies: These entities manage vast forest areas and seek efficient, scalable tech solutions for sustainable forest management, including health monitoring, pest and disease management, and illegal logging prevention. They face challenges like resource limitations and the need for accurate, timely data. These organizations prioritize investments in technology that enhances operational efficiency, supports sustainability, and provides environmental conservation value.
            ...
            """
            return f"{instructions}\n{example} Make sure to output {n} problems. OUTPUT\n"
        
        queries = [_make_prompt(query) for query in queries]
        return await super().generate(queries, model)


class SolutionGeneratorAsync(GeneratorAsync):
    def __init__(self, model: str = None, debug: bool = False):
        super().__init__(Solutions, system_prompt="You are designed to assist venture capitlaists in market research. Be concise and to the point, but include all relevant information.", model=model, debug=debug)

    async def generate(self, problems: List[str], model: str = None):
        def _make_prompt(problem: str):
            instructions = """
            INSRUCTIONS:
                For each sub problem, provide:
                    - the problem
                    - a list of solution names and their details

            SCHEMA:
                {
                    "solutions": [
                        {
                        "problem": "Example Problem",
                        "solutions_dict": [
                            {"Solution Name": "Solution Detail"}
                        ]
                        }...
                    ]
                }
            """
            input_data = f"INPUT DATA:\n{problem}"
            example = """
            EXAMPLE RESPONSE:

            {
                "solutions": [
                    {
                        "problem": "Pests and diseases contribute to deforestion"
                        "solutions_dict": [
                            {"Biological Control": "Using natural predators to control pests..."},
                            {"Genetic Resistance": "Developing trees that are resistant to pests..."},
                            {"Chemical Control": "Using pesticides to control pests..."},
                            {"Cultural Control": "Using traditional farming methods to control pests..."}
                            ...
                        ]
                    }...
                ]
            }
            """
            query = f"{instructions}\n{input_data}\n{example}\nOUTPUT:"
            return query
        queries = [_make_prompt(problem) for problem in problems]
        return await super().generate(queries, model)