from typing import List, Union
from lib.generators import GeneratorAsync
from autorest.models import GeneratedPydanticModel, RegeneratedCode, GeneratedDatabaseRepository
from autorest.examples import database_model_example, pydantic_model_example, database_repositoriy_example

class PydanticModelGenerator(GeneratorAsync):
    def __init__(self, model: str = None, debug: bool = False):
        super().__init__(
            GeneratedPydanticModel,
            system_prompt="You are designed to take in user instructons and answher the user's request concisely with only code and nothing but code.",
            model=model,
            debug=debug,
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

            Formatting Rules:
            - If a field can be missing from the database table without breaking the schema, use the Optional[] type hint in the Pydantic model.

            Import Rules:
            - Always include every necessary import.
            - You can combine imports on the same line if they are from the same module.
            - Import the UUID type as follows 'from uuid import UUID'
            - Import ConfigDict as follows 'from pydantic import ConfigDict'

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