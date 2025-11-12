import sys

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from bytez import Bytez

# Initialize Bytez SDK with your key
sdk = Bytez("ab84ea95bed7cae9a5ed19420b69e47f")

# Load the model via Bytez (using a free OpenAI model)
model = sdk.model("openai/gpt-4o-mini")

query = sys.argv[1] if len(sys.argv) > 1 else "default query"

# Load documents from ./data directory
loader = DirectoryLoader('./data', glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()

# Concatenate document contents as context
context = "\n".join([doc.page_content for doc in documents])

# Run the query with context using Bytez model
# Bytez expects a list of messages
messages = [
    {"role": "system", "content": "You are a helpful assistant. Answer based on the provided context."},
    {"role": "user", "content": f"Query: {query}\n\nContext: {context}\n\nAnswer the query based on the context."}
]
result = model.run(messages)
print(result)
