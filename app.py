import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from dotenv import load_dotenv, find_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import FAISS
# from langchain.llms import OpenAI
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import pickle
from PIL import Image

with st.sidebar:
    st.title("LLM Chat App")
    st.markdown('''
                ##About
                this app is build using:
                - [Streamlit](https://streamlit.io/)
                - [LangChain](https://python.langchain.com/)
                - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
                ''')
    add_vertical_space(5)
    st.write('Made by [Ihsan](https://github.com/ihsan292292/Chat_With_PDF_using-Langchain)')
    
load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

def main():
    image = Image.open('chatPDF.png')
    st.image(image,width=350)
    pdf = st.file_uploader("Upload Your File", type='pdf')
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        
        # text split
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        # st.write(chunks)
        
        # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = HuggingFaceHubEmbeddings()
            VectorStore = FAISS.from_texts(chunks,embedding=embeddings)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(VectorStore, f)
                
        # Accept user questions/query 
        query = st.text_input("Ask question about you PDF file:")
        
        if query:
            docs = VectorStore.similarity_search(query=query, k=2)
            repo_id = "tiiuae/falcon-7b-instruct"
            llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 500})
            chain = load_qa_chain(llm=llm,chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)
    
if __name__ == '__main__':
    
    main()