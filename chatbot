import os
import boto3
import json
from atlassian import Confluence
from langchain.agents.agent_types import AgentType
from langchain.chat_models import bedrock
from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.document_loaders import ConfluenceLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from Config import Config
import streamlit as st

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

class LangchainConfluenceLoader:

    def __init__(self):
        self.loader = ConfluenceLoader(
                                url=Config.confluence_url,
                                username=Config.confluence_username,
                                api_key=Config.confluence_api_key,
                               )

        bedrock_runtime = boto3.client(
                                service_name="bedrock-runtime",
                                region_name=Config.aws_region_name,
                                aws_access_key_id=Config.aws_access_key_id,
                                aws_secret_access_key=Config.aws_secret_access_key
        )

        # Create the Bedrock model
        self.bedrock = Bedrock(client=bedrock_runtime, model_id=Config.aws_llm_model_id)
        self.bedrock.model_kwargs = {"temperature": Config.llm_temperature,
                                     "max_tokens_to_sample": Config.llm_max_tokens_to_sample}



    def save_docs_to_vectorstore(self):

        documents = self.loader.load(space_key="KB", include_attachments=True, limit=50)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        #docsearch = PineconeVectorStore.from_documents(docs, self.embeddings, index_name="chatbot")
        db = Chroma.from_documents(docs, embeddings, persist_directory="chroma_db/")


    def search_docs_in_vectorstore(self, question):
        #vectorstore = PineconeVectorStore(index_name= "chatbot", embedding=self.embeddings)
        # save to disk
        db1 = Chroma(persist_directory="chroma_db/", embedding_function=embeddings)
        qa = RetrievalQA.from_chain_type(
            llm=self.bedrock,
            chain_type="stuff",
            retriever=db1.as_retriever()
        )
        return qa.invoke(question)

class CheckConfluenceUpdate:

    def __init__(self):

        self.confluence = Confluence(
                                url=Config.confluence_url,
                                username=Config.confluence_username,
                                password=Config.confluence_api_key,
                               )


    def fetch_page_id_in_space(self):
        # fetch all page ids in the space
        res = self.confluence.get_all_pages_from_space("KB", start=0, limit=100, status=None, expand=None, content_type='page')
        page_ids = [i['id'] for i in res]
        return page_ids


    def create_pageidLink_reference(self, page_ids):
        # creates a dct as follows page_id:link and dumps to json
        dct = {}
        for page_id in page_ids:
            page_history = self.confluence.history(page_id)
            dct[page_id] = page_history['lastUpdated']['_links']['self']
        with open('data.json', 'w') as f:
            json.dump(dct, f)
        qa = LangchainConfluenceLoader()
        qa.save_docs_to_vectorstore()


    def check_page_updates(self, page_ids, dct):
        for page_id in page_ids:
            # check if new page added
            if page_id not in dct:
                self.create_pageidLink_reference(page_ids)
                qa = LangchainConfluenceLoader()
                qa.save_docs_to_vectorstore()
                print("saved docs to vectorstore1")
                break

            # check if any existing page updated
            page_history = self.confluence.history(page_id)
            link = page_history['lastUpdated']['_links']['self']
            if link != dct[page_id]:
                pass
                qa = LangchainConfluenceLoader()
                qa.save_docs_to_vectorstore()
                print("saved docs to vectorstore2")
                break



if __name__ == "__main__":
    conf = CheckConfluenceUpdate()
    qa = LangchainConfluenceLoader()
    page_ids = conf.fetch_page_id_in_space()
    if not os.path.isfile('data.json'):
        conf.create_pageidLink_reference(page_ids)
    else:
        with open('data.json', 'r') as f:
            dct = json.load(f)
        conf.check_page_updates(page_ids, dct)
        
    page_ids = conf.fetch_page_id_in_space()
    if not os.path.isfile('data.json'):
        conf.create_pageidLink_reference(page_ids)
    else:
        with open('data.json', 'r') as f:
            dct = json.load(f)
        conf.check_page_updates(page_ids, dct)

    st.title("Chat with your Confluence!!")

    # Initialize session state for generated responses and past messages
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there!"]

    st.write("Please ask a question!")
    question = st.text_input("", key="input")
    st.session_state["past"].append(question)
    prompt = "Answer the query only in the context of text provided. Do not use your existing knowledge to answer the question. If you do not know the answer respond with 'I do not know the answer' "
    st.write("Query is: ", question)
    answer = qa.search_docs_in_vectorstore(prompt + question)
    st.write("Answer:", answer['result'])
        
