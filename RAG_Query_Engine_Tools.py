from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.agent import ReActAgent

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
llm = Ollama(model="llama3", request_timeout=120.0)

apple_index = None
uber_index = None
microsoft_index = None

try:
    storage_context = StorageContext.from_defaults(
        persist_dir="/Users/vishvassingh/Documents/Study/ML_AI/LlamaIndex/Storage/apple"
    )
    apple_index = load_index_from_storage(storage_context)

    storage_context = StorageContext.from_defaults(
        persist_dir="/Users/vishvassingh/Documents/Study/ML_AI/LlamaIndex/Storage/uber"
    )
    uber_index = load_index_from_storage(storage_context)

    storage_context = StorageContext.from_defaults(
        persist_dir="/Users/vishvassingh/Documents/Study/ML_AI/LlamaIndex/Storage/microsoft"
    )
    microsoft_index = load_index_from_storage(storage_context)

    index_loaded = True
except Exception as e:
    print(e)
    index_loaded = False

if not index_loaded:
    # load data
    apple_docs = SimpleDirectoryReader(
        input_files=["../Data/APPLE_RAG.pdf"]
    ).load_data()
    uber_docs = SimpleDirectoryReader(input_files=["../Data/UBER_RAG.pdf"]).load_data()
    microsoft_docs = SimpleDirectoryReader(
        input_files=["../Data/MICROSOFT_RAG.pdf"]
    ).load_data()

    embed_model = Settings.embed_model
    # build index
    apple_index = VectorStoreIndex.from_documents(apple_docs, embed_model=embed_model)
    uber_index = VectorStoreIndex.from_documents(uber_docs, embed_model=embed_model)
    microsoft_index = VectorStoreIndex.from_documents(
        microsoft_docs, embed_model=embed_model
    )
    # persist index
    apple_index.storage_context.persist(persist_dir="../Storage/apple")
    uber_index.storage_context.persist(persist_dir="../Storage/uber")
    microsoft_index.storage_context.persist(persist_dir="../Storage/microsoft")

    print("loaded data successfully")


apple_engine = apple_index.as_query_engine(similarity_top_k=3, llm=llm)
uber_engine = uber_index.as_query_engine(similarity_top_k=3, llm=llm)
microsoft_engine = microsoft_index.as_query_engine(similarity_top_k=3, llm=llm)


query_engine_tools = [
    QueryEngineTool(
        query_engine=apple_engine,
        metadata=ToolMetadata(
            name="apple_8k",
            description=(
                "Provides information about apple financials for year 2024"
                "Use a detailed plain text question as input to the tool"
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=uber_engine,
        metadata=ToolMetadata(
            name="uber_8k",
            description=(
                "Provides information about uber financials for year 2024"
                "Use a detailed plain text question as input to the tool"
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=microsoft_engine,
        metadata=ToolMetadata(
            name="microsoft_8k",
            description=(
                "Provides information about microsoft financials for year 2024"
                "Use a detailed plain text question as input to the tool"
            ),
        ),
    ),
]

agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True)

response = agent.chat("Compare the revenue growth of Microsoft and Apple in 2024")

print(response)
