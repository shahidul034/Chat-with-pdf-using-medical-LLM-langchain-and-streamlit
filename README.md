# Chat-with-pdf-using-LLM-langchain-and-streamlit
### Install those libraries
```bash
conda create -n lang python=3.8
conda activate lang
conda install -c conda-forge cudatoolkit=11.8.0
pip install nvidia-cudnn-cu11==8.6.0.163
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
python -m pip install jupyter
pip install langchain
pip install streamlit
pip install streamlit-chat
```
### Import libary
```
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from typing import Set
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from streamlit_chat import message
from typing import Any, List, Dict
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import textwrap
from transformers import pipeline
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
```

### Read the file path so that we can chat with LLM using this file. 

```
def get_file_path(uploaded_file):
    cwd = os.getcwd()
    temp_dir = os.path.join(cwd, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

f = st.file_uploader("Upload a file", type=(["pdf"]))
if f is not None:
    path_in = get_file_path(f)
    print("*"*10,path_in)
else:
    path_in = None
```
### This below line helps you to take the prompt for the LLM. It is the input to pass the LLM for inference.
```
prompt = st.text_input("Prompt", placeholder="Enter your prompt here..")
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
```
### Here, we take the LLM model locally, or you can take the model from the huggingface website.

```
if "model" not in st.session_state:
    # path=r"/home/drmohammad/data/llm/falcon-7b-instruct"
    # path=r"/home/drmohammad/data/llm/Llama-2-7b"
    path=r'/home/drmohammad/Documents/LLM/Llamav2hf/Llama-2-7b-chat-hf'
    tokenizer = AutoTokenizer.from_pretrained(path,
                                          use_auth_token=True,)

    model = AutoModelForCausalLM.from_pretrained(path,
                                             device_map='auto',
                                             torch_dtype=torch.float16,
                                             use_auth_token=True,
                                            #  load_in_8bit=True,
                                             load_in_4bit=True
                                             )

    pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens = 512,
                do_sample=True,
                top_k=30,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )
    
    llm = HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':0})
    st.session_state["model"] = llm
```
### We need to vectorize our pdf file to pass the data to LLM. If the pdf size crosses the LLM context length, it creates an error. We send only relevant chunks to LLM so that it never crosses the token limit. Here, We used huggingface embeddings. We can use OpenAI embeddings, but it needs to API call, which is not free.
```
if "vectorstore" not in st.session_state and path_in:
    loader=PyPDFLoader(file_path=path_in)
    documents=loader.load()
    text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=30,separator="\n")
    docs=text_splitter.split_documents(documents=documents)
    # embeddings=OpenAIEmbeddings()
    model_name = "sentence-transformers/all-mpnet-base-v2"
    hf = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore=FAISS.from_documents(docs,hf)
    vectorstore.save_local('langchain_pyloader/vectorize')
    new_vectorstore=FAISS.load_local("langchain_pyloader/vectorize",hf)
    print("pdf read done and vectorize")
     
    st.session_state["vectorstore"] = new_vectorstore
```
