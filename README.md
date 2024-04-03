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


## Changelog

### Week of 3/27
- [X] Update the generator code to use Anthropic
- [X] add reranking to the index class
- [X] Implement the multiple query inside the index by searching all queries, then removing duplicates, before finally reranking

### Week of 4/27
- [X] finish the Cohere class in `chat.py`
- [X] finish the Chat factory class
- [x] add tests for Chat factory class

## TODO
- [ ] sanity check all the new abstractions (see `notebooks/tests/sanity-check-chat.ipynb`)
- [ ] add tests to all `chat.py` abstractions
- [X] add ask_stream function to the abstract chat class
- [ ] figure out auto nest_asyncio
- [ ] figure out how you want to test this repo - pytest probably
- [ ] build and text the ability to generate queries that include the best location/db to search for
- [ ] Add tool calls for openai and Anthropic
- [ ] add Instructor support for Chat models
- [ ] move the models object from `utils.py` and to a .json config file 