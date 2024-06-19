# Chat-with-pdf-using-LLM-langchain-and-streamlit
🚀Custom finetuned model on medical data:  https://huggingface.co/shahidul034/Medical_Llama_2

![Interface](https://github.com/shahidul034/Chat-with-pdf-using-LLM-langchain-and-streamlit/blob/main/image.png)
### ⚙️Install those libraries
```bash
conda create -n lang python=3.8
conda activate lang
conda install -c conda-forge cudatoolkit=11.8.0
pip install nvidia-cudnn-cu11==8.6.0.163
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
pip install 'transformers[torch]'
pip install autotrain-advanced
python -m pip install jupyter
pip install langchain
pip install streamlit
pip install streamlit-chat
```
### 🎯Import libary
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

> 🎯Read the file path so that we can chat with LLM using this file. 

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
> 🎯This below line helps you to take the prompt for the LLM. It is the input to pass the LLM for inference.
```
prompt = st.text_input("Prompt", placeholder="Enter your prompt here..")
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
```
> 🎯Here, we take the LLM model locally, or you can take the model from the huggingface website. We load the model in 4-bit quantization so that we can run our model on a low-configuration PC (GPU: 12GB for inference).

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
> 🎯In order to effectively utilize our PDF data with a Large Language Model (LLM), it is essential to vectorize the content of the PDF. Given the constraints imposed by the LLM's context length, it is crucial to ensure that the data provided does not exceed this limit to prevent errors. To achieve this, we employ a process of converting the entire PDF file into vector chunks and only sending the relevant chunks to the LLM, thereby avoiding any issues related to token limits. To vectorize the PDF content, we utilize the "sentence-transformers/all-mpnet-base-v2" model, a state-of-the-art open-source vector embedding model. Additionally, we leverage FAISS (Facebook AI Similarity Search), an open-source library that facilitates efficient similarity search and clustering of dense vectors. This combination allows us to accurately and efficiently convert PDF content into vector embeddings. The vector embeddings are then saved locally, which not only optimizes the process by saving time during repeated use but also ensures that we do not incur additional costs associated with API calls. Although we have the option to use OpenAI embeddings, this approach requires API calls that are not free, making the use of open-source alternatives more practical and cost-effective for our purposes.
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
> 🎯This session_state helps to store the session input and output.
```
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

if "chat_history" not in st.session_state:
```
> 🎯We have developed a conversational LLMChain designed to process the vectorized output of PDF files. This LLMChain integrates memory capabilities, allowing it to retain input history and effectively pass this context to the LLM.
```
qa = ConversationalRetrievalChain.from_llm(
       llm=llm, retriever=new_vectorstore.as_retriever()
    )
res=qa({"question": query, "chat_history":chat_history})
```
