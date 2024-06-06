import streamlit as st
import pandas as pd
from io import StringIO
import tempfile
import langchain_helper
import utils

st.title("ChatDOC Demo using LangChain, Ollama, RAG and Gemma")

st.image("https://i.ytimg.com/vi/2k3cEiHag0M/maxresdefault.jpg", "chatdoc")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])

class ChatDoc:

    @utils.enable_chat_history
    def main(self):
        if uploaded_file is not None:
            file_type = uploaded_file.type
            print("file_type",file_type)
            if file_type == "application/pdf":
                pass

            else:
                print(uploaded_file)
                # To read file as bytes:
                bytes_data = uploaded_file.getvalue()
                # print(bytes_data)

                # To convert to a string based IO:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

                # To read file as string:
                tmp_file_path = ""
                string_data = stringio.read()
                # with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                #     tmp_file.write(bytes_data)
                #     tmp_file_path = tmp_file.name
                with open('example.txt', 'wb') as file:
                    file.write(bytes_data)
                    tmp_file_path = file.name

                print("tmp_file_path",tmp_file_path)
                # Use the temporary file path in the helper function

                text_document_txt = langchain_helper.data_ingestion_for_txt_file(tmp_file_path)
                # st.write(processed_data)
                # st.write(string_data)


                prompt = st.chat_input("Ask me anything!")
                if prompt:
                    utils.display_msg(prompt, 'user')
                    result = langchain_helper.main(text_document_txt, prompt)
                    utils.display_msg(result, 'assistant')
                    # st.write(result)



if __name__ == "__main__":
    obj = ChatDoc()
    obj.main()

