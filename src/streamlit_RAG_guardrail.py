import streamlit as st
from snowflake.snowpark.context import get_active_session
from snowflake.cortex import Complete
from snowflake.core import Root
from dotenv import load_dotenv
from snowflake.connector import connect
from trulens.core import TruSession
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.apps.custom import instrument, TruCustomApp
from trulens.core import Feedback, Select
from trulens.providers.cortex.provider import Cortex
import os
import pandas as pd
import json
import numpy as np

# Default Values
NUM_CHUNKS = 3
slide_window = 7

from snowflake.snowpark import Session
import json

# Load environment variables from .env file
load_dotenv(dotenv_path="src/.env")


# Load connection parameters from environment variables
connection_params = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA")
}

# Fetch the Cortex Search Service name
cortex_search_service_name = os.getenv("SNOWFLAKE_CORTEX_SEARCH_SERVICE")

# Validate all required environment variables are set
for key, value in connection_params.items():
    if not value:
        raise ValueError(f"Environment variable '{key}' is not set.")


# Define connection parameters
    
# Create a Snowpark session
snowpark_session = Session.builder.configs(connection_params).create()

# Use the session directly for operations
result = snowpark_session.sql("SELECT CURRENT_DATE;").collect()
print(result)

# # Service Parameters
CORTEX_SEARCH_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
CORTEX_SEARCH_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")
CORTEX_SEARCH_SERVICE = os.getenv("SNOWFLAKE_CORTEX_SEARCH_SERVICE")

# Columns to Query in the Service
COLUMNS = ["chunk", "relative_path", "category"]

# Load environment variables
load_dotenv(dotenv_path=".env")

# Initialize Snowflake session
session = snowpark_session
root = Root(session)

# Initialize TruLens and Snowflake connector
snowpark_session = session
tru_snowflake_connector = SnowflakeConnector(snowpark_session=snowpark_session)
tru_session = TruSession(connector=tru_snowflake_connector)
provider = Cortex(snowpark_session, "mistral-large")

# Feedback Functions
f_groundedness = Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness").on(
    Select.RecordCalls.retrieve_context.rets[:].collect()
).on_output()

f_context_relevance = Feedback(provider.context_relevance, name="Context Relevance").on_input().on(
    Select.RecordCalls.retrieve_context.rets[:]
).aggregate(np.mean)

f_answer_relevance = Feedback(provider.relevance, name="Answer Relevance").on_input().on_output().aggregate(np.mean)

feedbacks = [f_context_relevance, f_answer_relevance, f_groundedness]

# Create Retriever and RAG Class
class CortexSearchRetriever:
    def __init__(self, session, limit_to_retrieve=4):
        self._session = session
        self._limit_to_retrieve = limit_to_retrieve

    def retrieve(self, query):
        cortex_search_service = (
            root
            .databases[CORTEX_SEARCH_DATABASE]
            .schemas[CORTEX_SEARCH_SCHEMA]
            .cortex_search_services[CORTEX_SEARCH_SERVICE]
        )
        resp = cortex_search_service.search(query=query, columns=["chunk"], limit=self._limit_to_retrieve)
        return [curr["chunk"] for curr in resp.results] if resp.results else []

class RAGFromScratch:
    def __init__(self):
        self.retriever = CortexSearchRetriever(snowpark_session, limit_to_retrieve=4)

    @instrument
    def retrieve_context(self, query):
        return self.retriever.retrieve(query)

    @instrument
    def generate_completion(self, query, context_str):
        prompt = f"""
          You are an expert assistant extracting information from context provided.
          Answer the question based on the context. Be concise and do not hallucinate.
          If you don’t have the information just say so.
          Context: {context_str}
          Question:
          {query}
          Answer:
        """
        return Complete("mistral-large", prompt)

    @instrument
    def query(self, query):
        context_str = self.retrieve_context(query)
        return self.generate_completion(query, context_str)

rag = RAGFromScratch()
tru_rag = TruCustomApp(rag, app_name="RAG", app_version="simple", feedbacks=feedbacks)

def main():
    st.title(":speech_balloon: Enhanced RAG Chat with Observability")
    st.write("This app integrates Snowflake Cortex and TruLens for RAG with feedback and observability.")

    # Configuration Options
    st.sidebar.header("Configuration")
    st.sidebar.selectbox('Select model:', ('mistral-large', 'llama3-8b', 'llama3-70b'), key="model_name")
    st.sidebar.checkbox('Remember chat history?', key="use_chat_history", value=True)

    # Document Listing
    st.write("Available documents for answering queries:")
    docs_available = session.sql("ls @docs").collect()
    document_names = [doc["name"] for doc in docs_available]
    st.dataframe(pd.DataFrame(document_names, columns=["Document Names"]))

    # Accept User Input
    if question := st.chat_input("What do you want to know about your products?"):
        with tru_rag as recording:
            try:
                response = rag.query(question)
                st.chat_message("assistant").markdown(response)
            except Exception as e:
                st.chat_message("assistant").markdown(f"Error: {e}")

    # Display Leaderboard
    st.write("### Feedback Leaderboard")
    leaderboard = tru_session.get_leaderboard()
    st.dataframe(leaderboard)

if __name__ == "__main__":
    main()
