from openai import OpenAI
import tiktoken
from dotenv import load_dotenv
import re
from scipy import spatial
import cohere
from typing import List
import asyncio
import pandas as pd

from lib.utils import load_env
from lib.generators import QueryGeneratorAsync

env = load_env(['OPENAI_API_KEY', 'COHERE_API_KEY'])
OPENAI_API_KEY = env['OPENAI_API_KEY']
COHERE_API_KEY = env['COHERE_API_KEY']

class Reranker:
    @staticmethod
    def rerank(query: str, documents: List[str], top_n, model: str = "rerank-english-v2.0"):
        try: res = cohere.Client(api_key=COHERE_API_KEY).rerank(model=model, query=query, documents=documents, top_n=top_n)
        except Exception as e: raise Exception(f"Failed to rerank: {e}")
        if not res.results or len(res.results) == 0: raise Exception(f"No results returned from reranker")
        docs_and_scores = [(documents[result.index], round(result.relevance_score, 2)) for result in res.results]
        docs_and_scores.sort(key=lambda x: x[1], reverse=True) # might be redundant but just to be sure
        return zip(*docs_and_scores) # this unpacks the list of tuples into two lists

class Index:
    def __init__(self, embedding_model: str = 'text-embedding-3-large', debug: bool = False):
        self.model = self._init_model(embedding_model) # set model
        self.encoding = self._init_encoding(embedding_model) # set encoding
        self.data = None
        self.client: OpenAI = self._init_client() # set client
        self.debug = debug

    def _init_model(self, model: str):
        AVAILIBLE_MODELS = ['text-embedding-3-large'] # all models
        DEFAULT_MODEL = AVAILIBLE_MODELS[0] # default model
        if model is None: return DEFAULT_MODEL # return default model if none is provided
        if model not in AVAILIBLE_MODELS: raise ValueError(f"Model {model} not in {AVAILIBLE_MODELS}") # raise error if model is not in availible models
        return model # return model

    def _init_encoding(self, model: str):
        try: encoding = tiktoken.encoding_for_model(model)
        except: encoding = tiktoken.encoding_for_model("gpt-4")
        return encoding
    
    def _init_client(self):
        return OpenAI(api_key=OPENAI_API_KEY)
    
    # ! Utils ==========================================================
    def _sanitize_text(self, text):
        sanitized_text = re.sub(r"[^\x00-\x7F]+", "", text)
        sanitized_text = sanitized_text.replace("\n", " ").strip() # Trim leading and trailing whitespace
        if not sanitized_text: return None
        return sanitized_text
    
    def count_tokens(self, text: str) -> int: return len((encoding:=tiktoken.encoding_for_model("gpt-4")).encode(text)) if text else 0
    def split_tokens(self, text: str, size: int):
        encoding = tiktoken.encoding_for_model("gpt-4")
        tokens = encoding.encode(text)
        for i in range(0, len(tokens), size):
            if i + size > len(tokens):
                yield encoding.decode(tokens[i:])
            else:
                yield encoding.decode(tokens[i:i+size])

    # ! Embeddings ==========================================================
    def embed(self, text_list: List[str]) -> List[str]:
        if not isinstance(text_list, list) or len(text_list) < 1: raise TypeError("text_list must be a list of strings with at least one element")
        sanitized_text_list = [self._sanitize_text(text).replace("\n", " ").strip() for text in text_list]
        response = self.client.embeddings.create(input=sanitized_text_list, model=self.model)
        return [item.embedding for item in response.data]
    
    def process_text(self, text: str, chunk_size: int = 1000, overwrite=False) -> List[str]:
        if self.data is not None and overwrite==False: raise Exception("Data already loaded. Pass overwrite=True to overwrite")
        if not isinstance(text, str): raise TypeError("text must be a string")
        self.chunks = list(self.split_tokens(text, chunk_size))
        self.embeddings = self.embed(self.chunks)
        self.data = pd.DataFrame({"text": self.chunks, "embedding": self.embeddings})
        return self
    
    # ! Search ==========================================================
    def strings_ranked_by_relatedness(self, query: str, relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y), top_n: int = 100) -> tuple[list[str], list[float]]:
        """Returns a list of strings and relatednesses, sorted from most related to least."""
        if not isinstance(query, str): raise TypeError("query must be a string")
        if self.data is None: raise Exception("No data loaded. Call `process_text()` or `load_data()` first")
        query_embedding_response = self.client.embeddings.create(model="text-embedding-3-large", input=query,)
        query_embedding = query_embedding_response.data[0].embedding
        strings_and_relatednesses = [(row["text"], relatedness_fn(query_embedding, row["embedding"])) for i, row in self.data.iterrows()]
        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        strings, relatednesses = zip(*strings_and_relatednesses)
        return strings[:top_n], relatednesses[:top_n] # its okay if top_n is greater than the number of strings because slicing will handle that
    
    def get(self, query: str, top_n: int = 100) -> tuple[list[str], list[float]]:
        
        # First do some checks
        if not isinstance(query, str): raise TypeError("query must be a string")
        if self.data is None: raise Exception("No data loaded. Call `process_text()` or `load_data()` first")

        # Now we need to generate better queries (users are dumb a lot of the time)
        query_generator = QueryGeneratorAsync()
        res = asyncio.run(query_generator.generate(query)) # this will return a list of results
        parsed = res[0].get_query_text_only() # returns a list but we only need the first element
        if self.debug:
            print(f"Generated queries: {parsed}")

        # Now we need to get the strings and relatednesses for each query, then remove duplicates to prepare for reranking
        all_strings = []
        for query_obj in parsed:
            strings, relatednesses = self.strings_ranked_by_relatedness(query_obj.text, top_n=top_n)
            all_strings.extend([string for string in strings if string not in all_strings]) # extend with non-duplicate strings

        return Reranker.rerank(query=query, documents=all_strings, top_n=top_n) # this will return strings and scores just like the strings_ranked_by_relatedness method