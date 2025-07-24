from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

MODEL_NAME = "llama2" # You can change this to any model available in Ollama
TEMPERATURE = 0.7

def get_chat_model(api_key=None):
    # api_key parameter kept for compatibility but not used
    return Ollama(model=MODEL_NAME, temperature=TEMPERATURE)

def ask_chat_model(chat_model, query: str):
    response = chat_model.invoke(query)
    return response