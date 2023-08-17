import streamlit as st

with st.sidebar:
    st.title("LLM Chat App")
    st.markdown('''
                ##About
                this app is build using:
                - [Streamlit](https://streamlit.io/)
                - [LangChain](https://python.langchain.com/)
                - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
                ''')
    st.add_vertical_space(5)
    st.write('Made with ❤️ by [Ihsan](https://youtube.com/@engineerprompt)')
                