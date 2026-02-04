import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader

# Load all PDFs from a directory

loader = DirectoryLoader('aats_documents/',

                         glob="**/*.pdf", loader_cls=PyPDFLoader)

documents = loader.load()

print(f"Loaded {len(documents)} pages")


from langchain_text_splitters import RecursiveCharacterTextSplitter

# Split aats_documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=1000, # ~750 words
  chunk_overlap=200, # Preserve context
  length_function=len
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create vector store
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    location=":memory:",  # Or use your existing 'client' object
    collection_name="aats_knowledge",
    force_recreate=True # Useful for debugging/re-running in memory
)

from langchain.tools import tool

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

from langchain.agents import create_agent


tools = [retrieve_context]
# If desired, specify custom instructions
prompt = (
    "You have access to a tool that retrieves context from a documents. "
    "Use the tool to help answer user queries."
)

# Get your key: at console.groq.com set in environment variables
# Wrap as a Chat Model
from langchain_groq import ChatGroq

model = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)
agent = create_agent(model, tools, system_prompt=prompt)


# Ask a question
query = "Who is eligible for a research scholarship?"

# The agent returns it's answer
result = agent.invoke({"messages": [{"role": "user", "content": query}]})
final_answer = result["messages"][-1].content

print(final_answer)
