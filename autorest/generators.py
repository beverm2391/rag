from typing import List, Union
from lib.generators import GeneratorAsync
from autorest.models import (
    GeneratedPydanticModel,
    RegeneratedCode,
    GeneratedDatabaseRepository,
    RouteGroup,
    GeneratedTest,
)
from autorest.examples import (
    database_model_example,
    pydantic_model_example,
    database_repositoriy_example,
    route_group_example,
    test_example,
)


class PydanticModelGenerator(GeneratorAsync):
    def __init__(self, model: str = None, debug: bool = False):
        super().__init__(
            GeneratedPydanticModel,
            system_prompt="You are designed to take in user instructons and answher the user's request concisely with only code and nothing but code.",
            model=model,
            debug=debug,
            max_tokens=4096,
        )

    async def generate(
        self, one_or_many_queries: Union[List[str], str], model: str = None
    ):
        def _make_prompt(user_insturctions: str):
            intructions = f"""
            INSTRUCITONS:
            Your job is to generate valid pydantic models according to the newest documentation based on the user's request.
            Always output valid code and nothing but code.
            You output imports separately from the rest of the code.
            Always follow the example provided in the output if you are unsure of anything.
            The output model should inlclude every field in the database model.

            Formatting Rules:
            - If a field can be missing from the database table without breaking the schema, use the Optional[] type hint in the Pydantic model.

            Import Rules:
            - Always include every necessary import.
            - You can combine imports on the same line if they are from the same module.
            - Import the UUID type as follows 'from uuid import UUID'
            - Import ConfigDict as follows 'from pydantic import ConfigDict'

            Make sure to output the full model with every field from the database model.
            """
            example = f"""
            EXAMPLE DB MODEL (Input)

            {database_model_example}
            
            EXAMPLE PYDANTIC MODEL (Output)

            {pydantic_model_example}
            """
            query = f"{intructions}\n{example}\nUSER INSTRUCTIONS:\n{user_insturctions}\nOUTPUT:"
            return query

        def _handle_input(one_or_many_queries: Union[List[str], str]):
            """A function to handle the user input and return the list of prompts to be passed to the generator"""
            if type(one_or_many_queries) == str:
                return [
                    _make_prompt(one_or_many_queries)
                ]  # if user passes a string (one query) (we must still reuturn a list even though its one query)
            elif type(one_or_many_queries == list):
                return [
                    _make_prompt(query) for query in one_or_many_queries
                ]  # if user passes a list of queries
            else:
                raise ValueError(
                    f"Expected str or list, got {type(one_or_many_queries)}"
                )

        queries = _handle_input(one_or_many_queries)
        return await super().generate(queries, model)


class DatabaseRepositoryGenerator(GeneratorAsync):
    def __init__(self, model: str = None, debug: bool = False):
        super().__init__(
            GeneratedDatabaseRepository,
            system_prompt="You are designed to take in user instructons and answher the user's request concisely with only code and nothing but code.",
            debug=debug,
            model=model,
        )

    async def generate(
        self, one_or_many_queries: Union[List[str], str], model: str = None
    ):
        def _make_prompt(user_insturctions: str):
            intructions = f"""
            INSTRUCITONS:
            Your job is to generate valid code according to the newest documentation based on the user's request.
            Always output valid code and nothing but code.
            You output imports separately from the rest of the code.
            Always follow the example provided in the output if you are unsure of anything.

            Formatting Rules:
            - If a field can be missing from the database table without breaking the schema, use the Optional[] type hint in the Pydantic model.

            Import Rules:
            - Always include every necessary import.
            - You can combine imports on the same line if they are from the same module.
            """
            example = f"""
            EXAMPLE DB REPOSITORY for Exposure (Output)

            {database_repositoriy_example}
            """
            query = f"{intructions}\n{example}\nUSER INSTRUCTIONS:\n{user_insturctions}\nOUTPUT:"
            return query

        def _handle_input(one_or_many_queries: Union[List[str], str]):
            """A function to handle the user input and return the list of prompts to be passed to the generator"""
            if type(one_or_many_queries) == str:
                return [
                    _make_prompt(one_or_many_queries)
                ]  # if user passes a string (one query) (we must still reuturn a list even though its one query)
            elif type(one_or_many_queries == list):
                return [
                    _make_prompt(query) for query in one_or_many_queries
                ]  # if user passes a list of queries
            else:
                raise ValueError(
                    f"Expected str or list, got {type(one_or_many_queries)}"
                )

        queries = _handle_input(one_or_many_queries)
        return await super().generate(queries, model)


class RouteGroupGenerator(GeneratorAsync):
    def __init__(self, model: str = None, debug: bool = False):
        super().__init__(
            RouteGroup,
            system_prompt="You are designed to take in user instructions and answer the user's request concisely with only code and nothing but code.",
            model=model,
            debug=debug,
        )

    async def generate(
        self, one_or_many_queries: Union[List[str], str], model: str = None
    ):
        def _make_prompt(user_instructions: str):
            instructions = f"""
            INSTRUCTIONS:
            Your job is to take in a list of Python functions and output a list of dictionaries containing the function's route, method, and code.
            Always output valid code and nothing but code.

            Formatting Rules:
            - The route should be a string.
            - The method should be a string.
            - The function should be a string.
            """
            example = f"""
            EXAMPLE QUERY for Exposure (Input)
            'make CRUD routes for the Exposure model, which inlcudes the following functions:
            - get multiple, get by id
            - create, update, delete'

            EXAMPLE ROPUTE GROUP for Exposure (Output)

            {route_group_example}
            """
            query = f"{instructions}\n{example}\nUSER INSTRUCTIONS:\n{user_instructions}\nOUTPUT:"
            return query

        def _handle_input(one_or_many_queries: Union[List[str], str]):
            """A function to handle the user input and return the list of prompts to be passed to the generator."""
            if isinstance(one_or_many_queries, str):
                return [_make_prompt(one_or_many_queries)]
            elif isinstance(one_or_many_queries, list):
                return [_make_prompt(query) for query in one_or_many_queries]
            else:
                raise ValueError(
                    f"Expected str or list, got {type(one_or_many_queries)}"
                )

        queries = _handle_input(one_or_many_queries)
        return await super().generate(queries, model)


class TestGenerator(GeneratorAsync):
    def __init__(self, model: str = None, debug: bool = False):
        super().__init__(
            GeneratedTest,
            system_prompt="You are designed to take in user instructions and answer the user's request concisely with only code and nothing but code.",
            model=model,
            debug=debug,
        )

    async def generate(
        self, one_or_many_queries: Union[List[str], str], model: str = None
    ):
        def _make_prompt(user_instructions: str):
            instructions = f"""
            INSTRUCTIONS:
            Your job is to take in some Python code of a Pydantic model, a database repository, a route group, and a params object with 
            url, example data, and example update data and output a test file for that code based on the example provided.

            Formatting Rules:
            - replce the primary key uuid in the example data with `str(uuid.uuid4())`
            """
            example = f"""
            EXAMPLE Input (Input)
            'write CRUD tests for the Injury model and routes'
            
            - example url
                - "/api/v1/injuries/"
            - example data 
                "id": "d3ee52c6-09b9-4a2a-8876-90ee9a29ccf0",
                "tekmir_harm_id": "301ec09d-ca25-47f4-abc9-4ecf05319c0c",
                "contact_id": "6c03dad0-9bac-42d7-b426-4b5c3f71dc2c",
                "discovery_of_injury_date": None,
                "discovery_of_injury_date_precision": None,
                "injury_date": None,
                "injury_date_precision": None,
                "substantiation_score": 90,
                "substantiation_score_time": "2024-04-25T18:39:26.298397",
            - example update data
                - 'substantiation_score' : 80 

            EXAMPLE TEST for Injury (Output)

            {test_example}
            """
            query = f"{instructions}\n{example}\nUSER INSTRUCTIONS:\n{user_instructions}\nOUTPUT:"
            return query

        def _handle_input(one_or_many_queries: Union[List[str], str]):
            """A function to handle the user input and return the list of prompts to be passed to the generator."""
            if isinstance(one_or_many_queries, str):
                return [_make_prompt(one_or_many_queries)]
            elif isinstance(one_or_many_queries, list):
                return [_make_prompt(query) for query in one_or_many_queries]
            else:
                raise ValueError(
                    f"Expected str or list, got {type(one_or_many_queries)}"
                )

        queries = _handle_input(one_or_many_queries)
        return await super().generate(queries, model)


class CodeRegenerator(GeneratorAsync):
    def __init__(self, model: str = None, debug: bool = False):
        super().__init__(
            RegeneratedCode,
            system_prompt="You are designed to take in user instructions and answer the user's request concisely with only code and nothing but code.",
            model=model,
            debug=debug,
        )

    async def generate(
        self, one_or_many_queries: Union[List[str], str], model: str = None
    ):
        def _make_prompt(user_instructions: str):
            instructions = f"""
            INSTRUCTIONS:
            Your job is to take in invalid code, brainstorm why it didn't work, fix it, and output only the fixed code.
            Always output valid code and nothing but code.

            YOU MUST NOT REMOVE ANY CORE FUNCTIONALITY like:
            - attributes
            - number of methods, classes
            - etc.

            STEPS:
            1. Identify the cause of the error.
            2. Fix the error.
                1. Check for common mistakes (e.g. missing colons, indentation errors, typos, etc.).
                2. Ensure all necessary imports are included.
                3. Make sure the code is formatted correctly and adheres to PEP 8 guidelines.
                4. Debug with this process:
                    1. What does the error message tell me that could be useful?
                    2. What is the purpose of the code?
                    3. Is there a method that might be deprecated?
                    4. Are there any syntax errors?
                    5. continue......
            3. Output only the fixed code.

            Formatting Rules:
            - Ensure all necessary imports are included.
            - Maintain consistent code formatting and adhere to PEP 8 guidelines.
            - Always include every necessary import.
            - You can combine imports on the same line if they are from the same module.
            """
            query = f"{instructions}\nUSER INSTRUCTIONS:\n{user_instructions}\nOUTPUT:"
            return query

        def _handle_input(one_or_many_queries: Union[List[str], str]):
            """A function to handle the user input and return the list of prompts to be passed to the generator."""
            if isinstance(one_or_many_queries, str):
                return [_make_prompt(one_or_many_queries)]
            elif isinstance(one_or_many_queries, list):
                return [_make_prompt(query) for query in one_or_many_queries]
            else:
                raise ValueError(
                    f"Expected str or list, got {type(one_or_many_queries)}"
                )

        queries = _handle_input(one_or_many_queries)
        return await super().generate(queries, model)
