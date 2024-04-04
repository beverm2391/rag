import requests

from lib.chat import Chat
from lib.data import Index
from lib.model_config import MODELS, DEFAULTS


class Rag:
    _models = MODELS
    _defaults = DEFAULTS

    def __init__(self, model_name: str, temperature: float = 0, max_tokens: int = 4096, system_prompt: str = None, instruction: str = None, debug: bool = False):
        self.model_name = model_name
        self.debug = debug
        self.index = Index(debug=debug)

        instruction = instruction if instruction else self._defaults["instruction"]
        system_prompt = system_prompt if system_prompt else self._defaults["system_prompt"] # revert to default if not provided

        self.instruction = instruction
        self._chat = Chat(model_name, temperature, max_tokens, system_prompt, debug) # THIS NEEDS TO BE PREFIXED WITH AN UNDERSCORE TO AVOID METHOD CONFLICTS

    def process_text(self, text: str, chunk_size: int = 1000, overwrite=False) -> Index:
        """Embed text and add to index."""
        if self.debug: print("Calling `index.process_text()`...")
        return self.index.process_text(text, chunk_size, overwrite)
    
    def get(self, query: str, top_n: int = 100) -> tuple[list[str], list[float]]:
        """Get top_n results for the query. Automatically reranks the results."""
        if self.index.is_empty(): raise Exception("Index is empty. Please call process_text() first.")
        if self.debug: print("Calling `index.get()` ...")
        return self.index.get(query, top_n)
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text."""
        return self.index.count_tokens(text)
    
    def make_prompt(self, query: str, top_n: int = None) -> str:
        """Make a prompt using rag to get relevant context for the query."""
        if self.index.is_empty(): raise Exception("Index is empty. Please call process_text() first.")
        if self.debug: print("Making prompt...")

        strings, scores = self.get(query, top_n)
        tokens_left = input_token_budget = self._chat.context_window - self._chat.max_tokens # calculate how many tokens we can use in the input

        contexts = ""
        for i, (string, score) in enumerate(zip(strings, scores)):
            if tokens_left < 0: break

            if i > 0: contexts += "\n" # add newline if not the first context
            context_tokens = self.count_tokens(string)

            if context_tokens > tokens_left: # if the context is too long, truncate it
                # TODO add the truncated context here
                break
            
            contexts += f"Context: {i}, Relevance Score: {score}, Data: {string}"
            tokens_left -= context_tokens

        a  = f"INSTRUCTION: {self.instruction}\n\n"
        b = f"RELEVANT CONTEXTS:\n{contexts}\n\n"
        c = f"QUERY: {query}\n\n"
        d = f"ANSWER:"
        final_prompt = a + b + c + d

        # logging and sanity checks
        if self.debug: print(f"FINAL PROMPT\n\n{final_prompt}\n\n")
        assert final_prompt, "Prompt is None. Please check the prompt generation logic."
        assert (tokens:=self.count_tokens(final_prompt)) <= input_token_budget, f"Prompt is {tokens} tokens which exceeds the limit of {input_token_budget} tokens (context_window: {self._chat.context_window}, max_tokens: {self._chat.max_tokens}). This means the logic in `make_prompt()` is incorrect. Please check it."

        return final_prompt
    
    def chat(self, query: str, top_n: int = None) -> str:
        """Chat with rag."""
        if self.debug: print("Calling `_chat.chat()` ...")
        prompt = self.make_prompt(query, top_n)
        return self._chat.chat(prompt)
    
    def chat_stream(self, query: str, top_n: int = None) -> str:
        """Chat with rag in a streaming fashion."""
        if self.debug: print("Calling `_chat.chat_stream()` ...")
        prompt = self.make_prompt(query, top_n)
        return self._chat.chat_stream(prompt)
    
    def print_stream(self, query: str, top_n: int = None, line_size: int = 22):
        """Print the chat_stream output."""
        if self.debug: print("Calling `_chat.print_stream()` ...")
        prompt = self.make_prompt(query, top_n)
        return self._chat.print_stream(prompt, line_size)