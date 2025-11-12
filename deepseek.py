import os
import sys

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from bytez import Bytez

# Initialize Bytez SDK
sdk = Bytez("ab84ea95bed7cae9a5ed19420b69e47f")

# os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"  # Replace if needed

query = sys.argv[1] if len(sys.argv) > 1 else "default query"

# Load documents
loader = DirectoryLoader('./data', glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()

# Create embeddings and vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Query the index
result = qa_chain.run(query)
print(result)
