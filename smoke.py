from llama_index.llms.ollama import Ollama

llm = Ollama(model='llama3', request_timeout=60.0)

response = llm.complete('Tell me a joke')

print(response)