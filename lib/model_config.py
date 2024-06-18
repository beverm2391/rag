MODELS = {
    # ? DOCS: https://platform.openai.com/docs/models
    "gpt-3.5-turbo": {
        "name": "gpt-3.5-turbo",
        "model_var": "gpt-3.5-turbo",
        "org": "openai",
        "context_window": 16385,
        "price_per_million_tokens": {
            "input": 0.5,
            "output": 1.5,
        }
    },
    "gpt-4": {
        "name": "gpt-4",
        "model_var": "gpt-4",
        "org": "openai",
        "context_window": 8192,
        "price_per_million_tokens": {
            "input": 30,
            "output": 60
        }
    },
    "gpt-4-turbo": {
        "name": "gpt-4-turbo",
        "model_var": "gpt-4-turbo-preview",
        "org": "openai",
        "context_window": 128000,
         "price_per_million_tokens": {
            "input": 10,
            "output": 30
        }
    },
    "gpt-4o": {
        "name": "gpt-4o",
        "model_var": "gpt-4o",
        "org": "openai",
        "context_window": 128000,
        "price_per_million_tokens": {
            "input": 10,
            "output": 30
        }
    },
    # ? DOCS: https://docs.anthropic.com/claude/docs/models-overview
    "claude-3-opus": {
        "name": "claude-3-opus",
        "model_var": "claude-3-opus-20240229",
        "org": "anthropic",
        "context_window": 200000,
        "price_per_million_tokens": {
            "input": 15,
            "output": 75,
        }
    },
    "claude-3-sonnet": {
        "name": "claude-3-sonnet",
        "model_var": "claude-3-sonnet-20240229",
        "org": "anthropic",
        "context_window": 200000,
        "price_per_million_tokens": {
            "input": 3,
            "output": 15,
        }
    },
    "claude-3-haiku": {
        "name": "claude-3-haiku",
        "model_var": "claude-3-haiku-20240307",
        "org": "anthropic",
        "context_window": 200000,
        "price_per_million_tokens": {
            "input": 0.25,
            "output": 1.25,
        }
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
        "price_per_million_tokens": {
            "input": 0.5,
            "output": 1.5,
        }
    },
    "command-r-plus": {
        "name": "command-r-plus",
        "model_var": "command-r-plus",
        "org": "cohere",
        "context_window": 128000,
        "price_per_million_tokens": {
            "input": 3,
            "output": 15,
        }
    },
}

DEFAULTS = {
    "instruction" : "Answer the query based on the context provided.",
    "system_prompt" : "You are a helpful assistend designed to answer questions based on the context provided.",
    "top_n" : 30,
    "model": "gpt-3.5-turbo",
    "max_tokens": 4000,
    "temperature": 0,
}
