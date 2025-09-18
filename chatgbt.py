import os
import sys
import constant
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.llms import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
import time
from langchain_community.vectorstores import Chroma
from openai import RateLimitError
os.environ["OPENAI_API_KEY"] = constant.APIKEY

query = sys.argv[1]

#loader = TextLoader('data.txt')
loader = DirectoryLoader(".", glob="*.txt")
embedding = OpenAIEmbeddings()

def create_index_with_retry(loader, embedding, max_retries=3, delay=5):
    for attempt in range(max_retries):
        try:
            index = VectorstoreIndexCreator(vectorstore_cls=Chroma, embedding=embedding).from_loaders([loader])
            return index
        except RateLimitError as e:
            print(f"Rate limit exceeded, retrying in {delay} seconds... (Attempt {attempt + 1} of {max_retries})")
            time.sleep(delay)
    raise Exception("Failed to create index due to repeated rate limit errors.")

index = create_index_with_retry(loader, embedding)

print(index.query(query))
