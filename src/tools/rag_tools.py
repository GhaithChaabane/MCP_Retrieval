"""
RAG tools for MCP server
"""

import logging
from typing import Any, Dict, List
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from mcp.types import TextContent

from .base import BaseTool

logger = logging.getLogger(__name__)


class RagTool(BaseTool):
    """Tool for retrieving documents from a Vector Database"""

    def __init__(self, base_url: str = "http://localhost:6333"):
        self.base_url = base_url
        self.endpoint = f"{base_url}"


    def get_name(self) -> str:
        return "rag_docs"

    def get_description(self) -> str:
        return """
        It performs an intelligent query on the RAG document database.
        This tool searches company documents and provides answers based on actual content, including references to source documents. Ideal for finding specific information, details about people, processes, products or any documented content.
        """
    def qdrant_ingestion(self,data_path:str="data/crisis_inventory_structure"):
        loader = DirectoryLoader(
            data_path,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        split_docs = text_splitter.split_documents(docs)
        embedding_model = OllamaEmbeddings(
            model="qwen3-embedding:0.6b",
            temperature=0,
            base_url="http://localhost:11434",
        )
        #Store in Qdrant (Vector Database)
        vector_store = QdrantVectorStore.from_documents(
            documents=split_docs,
            url="http://localhost:6333",
            collection_name="mcp_collection",
            embedding=embedding_model,
        )
        return vector_store.collection_name

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question to ask the RAG system",
                }
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs) -> List[TextContent]:
        """Execute the RAG query with document information"""
        query = kwargs.get("query")
        if not query:
            return [
                TextContent(type="text", text="Error: Query parameter is required")
            ]
        
            

        embedding_model = OllamaEmbeddings(
                model="qwen3-embedding:0.6b",
                temperature=0,
                base_url="http://localhost:11434",
            )
        retriever = QdrantVectorStore.from_existing_collection(
                url="http://localhost:6333",
                collection_name="mcp_collection",
                embedding=embedding_model,
            )

        relevant_chunks = retriever.similarity_search(query)
        return {
                                    "success": True,
                                    "message": "Query executed successfully",
                                    "data": relevant_chunks,
                                    "query": query,
                                }

    