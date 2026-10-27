"""
RAG MCP Server
"""

import asyncio
import json
import logging
from typing import Any, Dict, List

from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer
from mcp.types import Resource, TextContent

# Import handling for both module and direct execution
try:
    from .tools.rag_tools import RagTool
except ImportError:
    # Fallback for direct execution
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tools.rag_tools import RagTool

# Load environment variables from .env file
load_dotenv()



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            "loggs/log.log",
            mode="w",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)


server = MCPServer("RAG MCP Server")

# Local RAG tools are initialized at startup; no API token is required.
AVAILABLE_TOOLS = []


def initialize_rag_tools(base_url: str = "http://localhost:6333"):
    """Initializes RAG tools with the provided parameters"""
    global AVAILABLE_TOOLS

    AVAILABLE_TOOLS=RagTool()
    logger.info("RAG tools initialized with base URL: %s", base_url)
    logger.info("RAG tools initialized with these tools")

    return AVAILABLE_TOOLS



async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handles tool calls"""
    global AVAILABLE_TOOLS
    logger.info("Handling tool call: %s with arguments: %s", name, arguments)
    if AVAILABLE_TOOLS is None or AVAILABLE_TOOLS == []:
        return [
            TextContent(
                type="text",
                text="RAG tools are not initialized.",
            )
        ]

    # if the tool is qdrant_ingestion
    collection_name=AVAILABLE_TOOLS.qdrant_ingestion(data_path="data/crisis_inventory_structure")
    logger.info("ingesting tools initialized with data_path")
    # if the tool is execute
    if name == "rag_docs":
        # if the tool is execute (for ivoking the querry)
        result = await AVAILABLE_TOOLS.execute(**arguments)

        # Format the result
        if isinstance(result, dict):
            if result.get("success", False):
                message = result.get("message", "Operation completed successfully")
                details = []

                # Specific handling for RAG results
                if "data" in result:
                    rag_data = result["data"]
                    if isinstance(rag_data, List):
                        # Extract RAG response
                        for i in rag_data:    
                            details.append(f"Response: {i.page_content}")

                        # Extract information about source documents
                        details.append("Source Documents:")
                        for i, source in enumerate(rag_data, 1):
                                if isinstance(source.metadata, dict):
                                    doc_info = f"  {i}. "
                                    if "source" in source.metadata:
                                        doc_info += f"Document: {source.metadata['source']}"
                                    if "score" in source.metadata:
                                        doc_info += f" (Score: {source.metadata['score']:.3f})"
                                    if "_collection_name" in source.metadata:
                                        doc_info += f"\n     collection_name: {source.metadata["_collection_name"]}"
                                    details.append(doc_info)

                if "query" in result:
                    details.append(f"Query: {result['query']}")

                # If there are no specific RAG details, use the base message
                if details:
                    full_message = "\n".join(details)
                else:
                    full_message = message

                return [TextContent(type="text", text=full_message)]
            else:
                error_message = result.get("error", "Unknown error")
                user_message = result.get("message", "Operation failed")
                return [
                    TextContent(
                        type="text",
                        text=f"{user_message}\n\nError details: {error_message}",
                    )
                ]
        else:
            return [TextContent(type="text", text="result not a dict")]



@server.tool(
    name="rag_docs",
    description="Performs an intelligent query on the local RAG document database.",
)
async def rag_docs(query: str) -> str:
    """Executes a RAG query through the existing tool handler."""
    result = await handle_call_tool("rag_docs", {"query": query})
    return result[0].text if result else ""


@server.resource(
    "rag://docs",
    name="RAG MCP Server Documentation",
    description="Documentation for using the RAG MCP server",
    mime_type="text/plain",
)
async def handle_read_resource() -> str:
    """Reads the content of a resource"""
    logger.info("Reading resource: rag://docs")
    return """
            # RAG MCP Server

            This MCP server provides access to RAG (Retrieval-Augmented Generation) features.

            ## Available Tools:

            ### rag_docs
            Executes a RAG query including information about source documents.
            Parameters:
            - query (string): The question to ask the RAG system

            ## Configuration:
            This server uses local Ollama embeddings and a local Qdrant instance.
            Ollama runs at http://localhost:11434 and Qdrant runs at http://localhost:6333.
            The server runs in stdio mode for MCP integration.

            ## Example usage:
            - Ask questions about your documents
            - Get contextual answers based on your vector database
            - Retrieve source document information when needed
        """


def main():
    """Main function to start the server"""
    try:
        logger.info("Starting RAG MCP server")

        initialize_rag_tools()
        logger.info("Starting STDIO server")
        server.run(transport="stdio")

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(
            "Unexpected error occurred while starting the server: %s",
            str(e),
            exc_info=True,
        )
        raise


def run_server():
    """Synchronous entry point for the console script"""
    main()


if __name__ == "__main__":
    run_server()
