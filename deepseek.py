import os
import sys
import getpass
from langchain_community.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma

if not os.environ.get("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("Enter API key for DeepSeek: ")

llm = init_chat_model("deepseek-chat", model_provider="deepseek")

query = sys.argv[1]

loader = DirectoryLoader(".", glob="*.txt")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

index = VectorstoreIndexCreator(vectorstore_cls=Chroma, embedding=embedding).from_loaders([loader])

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=index.vectorstore.as_retriever())

print(qa.invoke({"query": query}))
