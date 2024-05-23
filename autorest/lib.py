import os
import ast
from typing import List, Tuple

from lib.utils import verbose_validate
from autorest.models import (
    DatabaseModels,
    Route,
    RouteGroup,
    RouteConfig,
    RegeneratedCode,
    GeneratedTest,
    GeneratedPydanticModel,
    GeneratedDatabaseRepository,
)
from autorest.generators import (
    CodeRegenerator,
    PydanticModelGenerator,
    DatabaseRepositoryGenerator,
    RouteGroupGenerator,
    TestGenerator,
)
from collections import namedtuple

# these must be outside of the class!
globals_ = globals()
locals_ = locals()


class DynamicOutputValidator:
    def __init__(self, code: str, retries: int = 2, model: str = 'gpt-4-turbo', debug: bool = False):
        self.code: List[str] = [code]
        self.retries = retries
        self.generator = CodeRegenerator(model=model)
        self.debug = debug
        self.result = namedtuple("Test", ["result", "code"])

    async def validate(self) -> Tuple[bool, str]:
        for i in range(self.retries):
            current_best_code = self.code[-1]
            if self.debug:
                print(f"Trying code:\n{current_best_code}")
            try:
                exec(current_best_code, globals_, locals_)
                print("Code is valid!")
                return self.result(True, current_best_code)
            except Exception as e:
                print("Code is invalid. Exception: ", e)
                print(f"Regenerating code... try {i+1} / {self.retries}")
                prompt = f"This error was thrown by the code below:\n{str(e)}\nCode:\n{current_best_code}"
                res: List[RegeneratedCode] = await self.generator.generate(prompt)
                new_best_code = res[0].code
                self.code.append(new_best_code)
        return self.result(False, current_best_code)


class GenerativeConfig:
    def __init__(self, root_dir: str = None, debug: bool = False):
        self.db_models: DatabaseModels = {}
        self.route_config: RouteConfig = {}
        self.root_dir: str = root_dir
        self.debug: bool = debug

    def _validate_path(self, path: str):
        if self.root_dir:
            print(f"joining relative path to root dir {self.root_dir}")
            path = os.path.join(self.root_dir, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        return path

    def get_db_models(self, db_models_path: str) -> dict:
        """
        Reads a Python file and returns a dictionary where keys are the names of the classes
        and values are the corresponding code blocks of those classes.

        :param db_models_path: Path to the Python file
        :return: Dictionary with class names as keys and code blocks as values
        """
        db_models_path = self._validate_path(db_models_path)

        with open(db_models_path, "r") as file:
            source_code = file.read()
            tree = ast.parse(source_code, filename=db_models_path)

        class_definitions = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                class_start_line = node.lineno - 1
                class_end_line = (
                    node.end_lineno if hasattr(node, "end_lineno") else None
                )

                if class_end_line is None:
                    # If `end_lineno` is not available, use a fallback method to determine the end line
                    class_end_line = len(source_code.splitlines())

                class_code = "\n".join(
                    source_code.splitlines()[class_start_line:class_end_line]
                )
                class_definitions[class_name] = class_code

        return class_definitions

    def get_route_group(self, path: str) -> dict:
        """
        Reads a Python file and returns a dictionary with route configurations.

        :param path: Path to the Python file
        :return: Dictionary with route configurations
        """
        path = self._validate_path(path)

        with open(path, "r") as file:
            source_code = file.read()
            tree = ast.parse(source_code, filename=path)

        routes = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call) and isinstance(
                        decorator.func, ast.Attribute
                    ):
                        if (
                            isinstance(decorator.func.value, ast.Name)
                            and decorator.func.value.id == "router"
                        ):
                            method = decorator.func.attr
                            path = decorator.args[0].s if decorator.args else ""

                            # Extract function code
                            function_start_line = node.lineno - 1
                            function_end_line = (
                                node.end_lineno if hasattr(node, "end_lineno") else None
                            )

                            if function_end_line is None:
                                # If `end_lineno` is not available, use a fallback method to determine the end line
                                function_end_line = len(source_code.splitlines())

                            function_code = "\n".join(
                                source_code.splitlines()[
                                    function_start_line:function_end_line
                                ]
                            )

                            route_ = {
                                "route": path,
                                "method": method.upper(),
                                "function": function_code,
                            }
                            routes.append(verbose_validate(Route, route_))

        rg = {"routes": routes}
        return verbose_validate(RouteGroup, rg)

    def get_route_config(self, paths: List[str]):
        if not isinstance(paths, list):
            raise TypeError("paths must be a list of strings to each route file.")
        paths = [self._validate_path(path) for path in paths]
        print(paths)
        route_config = {
            (basename := os.path.basename(path).split(".")[0]): self.get_route_group(
                path
            )
            for path in paths
        }
        return verbose_validate(RouteConfig, route_config)

    def set_db_models(self, path: str):
        self.db_models = self.get_db_models(path)

    def set_route_group(self, path: str):
        self.route_group = self.get_route_group(path)

    def set_route_config(self, paths: List[str]):
        self.route_config = self.get_route_config(paths)


class AutoRest:
    def __init__(
        self,
        config: GenerativeConfig,
        table: str,
        model: str = "gpt-4-turbo",
        debug: bool = False,
    ):
        self.config: GenerativeConfig = self._init_config(config)
        self.table = table
        self.pydantic_model: str = None
        self.db_repo: GeneratedDatabaseRepository = None
        self.route_group: RouteGroup = None
        self.tests = None
        self.model: str = model
        self.debug: bool = debug

    def _init_config(self, config: GenerativeConfig):
        if not isinstance(config, GenerativeConfig):
            raise TypeError("config must be an instance of GenerativeConfig")
        if not config.db_models:
            raise Exception(
                "No database models found. Run `set_db_models(path)` first."
            )
        # if not config.route_config:
        #     raise Exception("No route configurations found. Run `set_route_config(paths)` first.")
        return config

    async def generate_pydantic_model(self):
        """Automatically generate Pydantic models based on the database models."""
        db_model_code: str = self.config.db_models.get(self.table, None)
        if not db_model_code:
            raise Exception("something went wrong")

        generator = PydanticModelGenerator(
            model=self.model, debug=self.debug
        )  # init the generator
        generated_pydantic_model: GeneratedPydanticModel = (
            await generator.generate(db_model_code)
        )[
            0
        ]  # generate the model (coroutine must be in parentheses to index it)
        imports, code = (
            generated_pydantic_model.imports,
            generated_pydantic_model.code,
        )  # extract the imports and code
        total_code = f"{imports}\n{code}"  # combine the imports and code for validation

        dynamic_output_validator = DynamicOutputValidator(
            total_code, debug=self.debug, retries=2, model=self.model
        )  # init the dynamic output validator
        is_valid, code = (
            await dynamic_output_validator.validate()
        )  # dynamically the code

        if is_valid:
            self.pydantic_model = code
        else:
            raise Exception("Could not generate valid code")

    async def generate_db_repo(self):
        """Automatically generate a database repository based on the database model."""
        generator = DatabaseRepositoryGenerator(
            model=self.model, debug=self.debug
        )  # init the generator
        generated_repo: GeneratedDatabaseRepository = (
            await generator.generate(self.table)
        )[0]
        self.db_repo = generated_repo

    async def generate_route_group(self):
        """Automatically generate a route group based on the pydantic model and the database repo."""
        generator = RouteGroupGenerator(model=self.model, debug=self.debug)
        prompt = f"""
        table: {self.table}
        Pydanctic Model: {self.pydantic_model}
        Database Repository: {self.db_repo}
        """
        generated_route_group: RouteGroup = (
            await generator.generate(prompt)
        )[0]
        self.route_group = generated_route_group

    async def generate_tests(self, name: str, url: str, example_data, example_update, instructions: str = None):
        """Automatically generate tests based on the route group."""
        generator = TestGenerator(model=self.model, debug=self.debug)
        prompt = f"""
        table: {self.table}
        Model Name: {name}
        Route Group: {self.route_group}
        URL: {url}
        Example Data: {example_data}
        Example Update: {example_update}
        User Instructions: {instructions if instructions else ""}
        """
        generated_tests: GeneratedTest = (await generator.generate(prompt))[0]
        self.tests = generated_tests

    def dump_route_group(self):
        routes = self.route_group.routes
        text = ""
        for route in routes:
            text += f"@router.{route.method}('{route.route}')\n"
            text += route.function
            text += "\n\n"
        
        return text

    def dump_to_files(self, output_dir: str) -> dict:
        file_basename = self.table.lower()
        file_paths = {
            "pydantic_model": os.path.join(output_dir, f"{file_basename}_model.py"),
            "db_repo": os.path.join(output_dir, f"{file_basename}_repo.py"),
            "route_group": os.path.join(output_dir, f"{file_basename}_routes.py"),
            "tests": os.path.join(output_dir, f"{file_basename}_tests.py"),
        }

        pydantic_model_full_code = self.pydantic_model
        db_repo_full_code = self.db_repo.imports + "\n\n" + self.db_repo.code
        route_group_full_code = self.dump_route_group()
        tests_full_code = self.tests.imports + "\n\n" + self.tests.code

        def save_to_file(file_path, content):
            with open(file_path, "w") as file:
                file.write(content)

        save_to_file(file_paths["pydantic_model"], pydantic_model_full_code)
        save_to_file(file_paths["db_repo"], db_repo_full_code)
        save_to_file(file_paths["route_group"], route_group_full_code)
        save_to_file(file_paths["tests"], tests_full_code)

        return file_paths