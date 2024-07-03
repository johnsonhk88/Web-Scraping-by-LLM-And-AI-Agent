from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI # if you want to use OpenAI for LLM Model
import os, time, json
from dotenv import load_dotenv
import torch 
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from langchain.prompts.prompt import PromptTemplate
from langchain_community.vectorstores import FAISS, Chroma

from langchain.embeddings import HuggingFaceEmbeddings
from datasets import Dataset, DatasetDict, load_dataset


USE_OPENAI = False
USE_GEMMA = True
USE_LLAMA3 = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

load_dotenv()
huggingfaceToken = os.getenv("HuggingFace") # for Open LLM Model Access token

# model configure
do_sample= True 
top_p=0.95 
top_k= 2
temperature=0.2#0.7 
num_beams = 3
max_length= 512

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
)


if USE_OPENAI:
    openaiKey = os.getenv("OPENAI_API_KEY")
    ModelName = "gpt-3.5-turbo"
    llmModel = ChatOpenAI(model=ModelName, api_key=openaiKey)

elif USE_GEMMA:
    ModelName = "google/gemma-2b-it" # or "google/gemma-7b-it"
    if device.type == "cuda":
        llmModel = AutoModelForCausalLM.from_pretrained(ModelName,
                                                         quantization_config=bnb_config, 
                                                         token=huggingfaceToken,
                                                          device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(ModelName, token=huggingfaceToken)
    else:
        llmModel = AutoModelForCausalLM.from_pretrained(ModelName, 
                                                        token=huggingfaceToken,
                                                        device_map="auto")
                                                        
        tokenizer = AutoTokenizer.from_pretrained(ModelName, token=huggingfaceToken)

elif USE_LLAMA3:
    ModelName = "meta-llama/Meta-Llama-3-8B" 
    if device.type == "cuda":
        llmModel = AutoModelForCausalLM.from_pretrained(ModelName, 
                                                        quantization_config=bnb_config, 
                                                        token=huggingfaceToken,
                                                        device_map="auto")
                                                        
        tokenizer = AutoTokenizer.from_pretrained(ModelName, token=huggingfaceToken)

    else:
        llmModel = AutoModelForCausalLM.from_pretrained(ModelName, 
                                                        token=huggingfaceToken,
                                                        device_map="auto")
                                                        
        tokenizer = AutoTokenizer.from_pretrained(ModelName, token=huggingfaceToken)


def getResponse(query, maxNewToken= 512):
    ids = tokenizer(query, return_tensors='pt').to(device)
    response = llmModel.generate(   **ids, 
                                    do_sample= True,
                                    top_p =0.95,
                                    top_k= 3,
                                    temperature= 0.2,
                                    max_new_tokens= maxNewToken,)

    return tokenizer.decode(response[0][len(ids["input_ids"]):], skip_special_tokens=True)


ret= getResponse("What is the Machine Learning?")
print(ret)

