# My Retrieval Augmented Generation (RAG) Implementations

## Goals

The first goal of this project is to build a robust RAG system that encapsulates the newest methods and models in an intuitive python interface. The second goal is to put all this logic on a server and expose REST APIs to a Next.js frontend that will include a Generative User Intergace. TBD

## Core 

This is a work in progress. Right now I'm finishing up the core abstractions. Everything is broken down into a few core services:

- `data.py`
  - this includes the `Index` class (embedding database)
  - and the `Reranker` class
- `generators.py`
  - this containes the generators that use [instructor]() to output schema-validated data
  - `models.py` contains the data models
- `chat.py`
  - this class wraps around popular LLM APIs and exposes certain methods for interface
- `rag.py`
  - TODO: this class will have an end to end configurable rag solution and leverage each of the aforementioned abstractions
- `utils.py`
  - pretty self explanatory, just utils and configuration

## Running Tests

- use `pytest` to run all local tests
- use `pytest --external` to run all local tests, including those that make external API calls (which can take a while)

## Changelog

### Week of 3/27
- [X] Update the generator code to use Anthropic
- [X] add reranking to the index class
- [X] Implement the multiple query inside the index by searching all queries, then removing duplicates, before finally reranking

### Week of 4/3
- [X] finish the Cohere class in `chat.py`
- [X] finish the Chat factory class
- [x] add tests for Chat factory class
- [X] add ask_stream function to the abstract chat class
- [X] sanity check all the new abstractions (see `notebooks/tests/sanity-check-chat.ipynb`)
  - [X] finish cohere stream, others are working
- [X] make a full RAG abstraction
- [X] get all `chat.py` tests passing
- [X] updated `chat_stream` and `print_stream` to yield/return mulitple response objects
- [X] got a FastAPI server running, configured globals, middleware, routes, and models
- [X] got the `/chat` endpoint working with OAI
- [X] figure out how to make the temp, model, and max_tokens params optional in `server/models/chat`
- [X] sanity check in a notebook the `/chat` endpoint with OAI, Anthropic, and Cohere
- [X] implement tests in `tests/server/routes/chat`
- [X] move server prints to logging?
- [X] get the streaming endpoint working
- [X] expose an all purpose chat endpoint that takes in params and returns a stream (FastAPI)
  
### Week of 4/30
- [X] rebuild logging to use one folder in the root dir
- [X] implement the message logic in the `chat` functions
  - [X] get all tests passing
  - [X] merge into main
- [X] added more robust error handling to the chat routes
- [X] added message validation rules for order
- [x] add usage/cost monitoring per request

## TODO
- [ ] integrate LlaMA 3 70B and 400B when it drops!! (with GROQ)
  - [ ] write in the docs the process for integraging a new model (add the specific class, add to `Chat` factory class, add message convert messages, add to model config)
- [ ] change the default RAG to persist=False
- [ ] add logging config for lib
- [ ] add usage printing per request/response in Debug mode
- [ ] add [rerank 3](https://twitter.com/aidangomez/status/1778416325628424339?s=42&t=xWkJ8jxhhZSSUcSQM9eZIQ)
- [ ] check multiple message support for all models (Claude is not working rn)
- [ ] build and text the ability to generate queries that include the best location/db to search for
- [ ] Add tool calls for openai and Anthropic
- [ ] add Instructor support for Chat models
- [ ] add rate limits to model config
- [ ] make model config into a data class, add embedding models and rerank models