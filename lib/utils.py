from dotenv import load_dotenv
import os
import asyncio
from pypdf import PdfReader
import re
from pptx import Presentation
import tiktoken
from functools import wraps

def load_env(expected_vars: list = []):
    env = load_dotenv()
    assert env, "No .env file found"
    vars = {}
    for var in expected_vars:
        assert os.getenv(var), f"Expected {var} in .env"
    return os.environ

def count_tokens(text: str, model: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text)) if text else 0

def pdf_to_text(fpath: str): return''.join([page.extract_text() for page in PdfReader(fpath).pages]).replace('\n', ' ').strip()

def clean_text(text):
    text = re.sub(r'\n\s*\n', '\n', text)
    return re.sub(r' {2,}', ' ', text).replace("\n", " ").strip()

def PPTX_to_text(path: str):
    """Extract the text from a PPTX file."""
    if not os.path.exists(path): raise FileNotFoundError(f"File {path} not found")
    
    prs = Presentation(path)
    text_runs = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    text_runs.append(run.text)
    return "\n".join(text_runs)

def auto_nest_asyncio(func):
    import nest_asyncio

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            print(f"Caught {e}", "\nThis is usually becasue a jupyter notebook has a running event loop. No problem - automatically nesting event loops...")
            if str(e) == 'asyncio.run() cannot be called from a running event loop':
                nest_asyncio.apply()
                return func(*args, **kwargs)
            else:
                raise e
    return wrapper