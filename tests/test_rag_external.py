import pytest
from functools import lru_cache
import requests

from lib.utils import validate_stream
from lib.rag import Rag

@pytest.fixture
@lru_cache #? cache this to avoid multiple requests
def shakespeare_text():
    return requests.get("https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt").text

@pytest.mark.external
def test_rag(shakespeare_text):
    
    rag = Rag("gpt-3.5-turbo", temperature=0, max_tokens=4096, system_prompt=None, instruction=None, debug=True)
    rag.process_text(shakespeare_text)
    res = rag.chat_stream("What is the meaning of life?")   
    validate_stream(res)