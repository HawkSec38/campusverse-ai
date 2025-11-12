import os
import sys
import constant
from bytez import Bytez
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma

sdk = Bytez(constant.APIKEY)

query = sys.argv[1]

loader = TextLoader('data.txt')
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

index = Chroma.from_documents(loader.load(), embedding)

# Assuming Bytez can be used as LLM in LangChain, but if not, this may need adjustment
# For now, using a placeholder; Bytez integration might require custom implementation
# llm = sdk.model("openai/gpt-4o-mini")  # Direct Bytez model
# Since RetrievalQA expects a LangChain LLM, you may need to wrap Bytez or use a compatible integration

# Placeholder for LLM using Bytez - this might not work directly; check Bytez docs for LangChain integration
# If BytezLLM exists in langchain_community, use it; otherwise, custom wrapper needed
try:
    from langchain_community.llms import BytezLLM
    llm = BytezLLM(model="openai/gpt-4o-mini", api_key=constant.APIKEY)
except ImportError:
    # Fallback or custom implementation
    print("BytezLLM not available; please install or implement custom LLM wrapper.")
    exit(1)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=index.vectorstore.as_retriever())

print(qa.invoke({"query": query}))
