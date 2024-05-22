import os
import ast
from typing import List, Tuple

from lib.utils import verbose_validate
from autorest.models import DatabaseModels, Route, RouteGroup, RouteConfig, RegeneratedCode
from autorest.generators import CodeRegenerator

class GenerativeConfig:
    def __init__(self, root_dir: str = None):
        self.db_models: DatabaseModels = {}
        self.route_config: RouteConfig = {}
        self.root_dir: str = root_dir

    def _validate_path(self, path: str):
        if self.root_dir:
            print(f"joining relative path to root dir {self.root_dir}")
            path = os.path.join(self.root_dir, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        return path
    
    def _validate_db_model(self, name: str):
        if not self.db_models:
            self.set_db_models()
        
        if name.lower() not in [k.lower() for k in self.db_models.keys()]:
            raise Exception(f"name {name} not found in db models")
    

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
        paths = [self._validate_path(path) for path in paths]
        route_config = {
            (basename := os.path.basename(path).split(".")[0]): self.get_route_group(
                path
            )
            for path in paths
        }
        return verbose_validate(RouteConfig, route_config)

    def set_db_models(self, *args, **kwargs):
        self.db_models = self.get_db_models(*args, **kwargs)

    def set_route_config(self, *args, **kwargs):
        self.route_config = self.get_route_config(*args, **kwargs)

    def make_pydantic_models(self, name: str):
        self._validate_db_model(name)
        # TODO do the generation
        db_model_code: str = self.db_models.get(name, None)
        if not db_model_code:
            raise Exception("something went wrong")

        generated_model = ""
        self.pydantic_model = generated_model

    def make_db_repo(self, name: str):
        self._validate_db_model(name)
        # TODO do the generation
        generated_repo = ""
        self.db_repo = generated_repo

    def make_route_group(self, name: str):
        self._validate_db_model(name)

        if not self.db_repo:
            self.make_db_repo(name)

        # TODO do the generation
        generated_route_group: RouteGroup = ""
        self.route_group = generated_route_group

    def make_tests(self, name: str):
        self._validate_db_model(name)

        if not self.route_group:
            self.make_route_group(name)

        # TODO do the generation
        generated_tests = ""
        self.tests = generated_tests


class DynamicOutputValidator:
    def __init__(self, code: str, retries: int = 2, debug: bool = False):
        self.code: List[str] = [code]
        self.retries = retries
        self.generator = CodeRegenerator(model='gpt-4-turbo')
        self.debug = debug

    async def regenerate(self) -> Tuple[bool, str]:
        for i in range(self.retries):
            current_best_code = self.code[-1]
            if self.debug:
                print(f"Trying code:\n{current_best_code}")
            try:
                exec(current_best_code)
                print("Code is valid!")
                return (True, current_best_code)
            except Exception as e:
                print("Code is invalid. Exception: ", e)
                print(f"Regenerating code... try {i+1} / {self.retries}")
                prompt = f"This error was thrown by the code below:\n{str(e)}\nCode:\n{current_best_code}"
                res: List[RegeneratedCode] = await self.generator.generate(prompt)
                new_best_code = res[0].code
                self.code.append(new_best_code)
        return (False, current_best_code)