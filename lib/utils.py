from dotenv import load_dotenv
import os
import asyncio
from PyPDF2 import PdfReader
import re
from pptx import Presentation
import tiktoken

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

MODELS = {
    # ? DOCS: https://platform.openai.com/docs/models
    "gpt-3.5-turbo": {
        "model_var": "gpt-3.5-turbo",
        "org": "openai",
        "context_window": 16385,
    },
    "gpt-4": {
        "name": "gpt-4",
        "model_var": "gpt-4",
        "org": "openai",
        "context_window": 8192,
    },
    "gpt-4-turbo": {
        "name": "gpt-4-turbo",
        "model_var": "gpt-4-turbo-preview",
        "org": "openai",
        "context_window": 128000,
    },
    # ? DOCS: https://docs.anthropic.com/claude/docs/models-overview
    "claude-3-opus": {
        "name": "claude-3-opus",
        "model_var": "claude-3-opus-20240229",
        "org": "anthropic",
        "context_window": 200000,
    },
    "claude-3-sonnet": {
        "name": "claude-3-sonnet",
        "model_var": "claude-3-sonnet-20240229",
        "org": "anthropic",
        "context_window": 200000,
    },
    "claude-3-haiku": {
        "name": "claude-3-haiku",
        "model_var": "claude-3-haiku-20240307",
        "org": "anthropic",
        "context_window": 200000,
    },
    # ? DOCS: https://docs.cohere.com/docs/models
    "command-light" : {
        "name": "command-light",
        "model_var": "command-light",
        "org": "cohere",
        "context_window": 4096,
    },
    "command-light-nightly" : {
        "name": "command-light-nightly",
        "model_var": "command-light-nightly",
        "org": "cohere",
        "context_window": 8192,
    },
    "command": {
        "name": "command",
        "model_var": "command",
        "org": "cohere",
        "context_window": 4096,
    },
    "command-nigihtly": {
        "name": "command-nightly",
        "model_var": "command-nightly",
        "org": "cohere",
        "context_window": 8192,
    },
    "command-r": {
        "name": "command-r",
        "model_var": "command-r",
        "org": "cohere",
        "context_window": 128000,
    },
    "command-r-plus": {
        "name": "command-r-plus",
        "model_var": "command-r-plus",
        "org": "cohere",
        "context_window": 128000,
    },
}
