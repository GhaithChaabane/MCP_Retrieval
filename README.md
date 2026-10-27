## HZS RAG MCP Server

An MCP server that provides document search over emergency and crisis-management information using Retrieval-Augmented Generation (RAG).

The server loads Markdown documents, creates embeddings using Ollama, stores them in Qdrant, and exposes a rag_docs MCP tool for querying the indexed content.

## Main Components

src/server.py — MCP server entrypoint and tool registration.

src/tools/rag_tools.py — ingestion and document-query logic.

src/tools/base.py — base interface for MCP tools.

data/crisis_inventory_structure/ — source Markdown documents.

Qdrant — vector database for document embeddings.

## Start Qdrant

docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  qdrant/qdrant

## Ollama — local embedding model provider.
Start Ollama

Make sure Ollama is running and the embedding model is available:

- ollama pull qwen3-embedding:0.6b

Ollama is expected at:

- http://localhost:11434

Qdrant is expected at:

- http://localhost:6333

## Install Dependencies

Create or use the project environment:

- uv sync