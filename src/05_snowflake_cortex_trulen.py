# %% [markdown]
# ## STEP 2: Query RAG with Snowflake Cortex and TruLens
# 
# By completing this guide, you'll get started with LLMOps by building a RAG by combining [Cortex LLM Functions](https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions) and [Cortex Search](https://github.com/Snowflake-Labs/cortex-search?tab=readme-ov-file), and then using [TruLens](https://www.trulens.org/) to add observability and guardrails.
# 
# We will use Step 1 created chunked text table and embedding 
# - Database: FINANC_CORTEX_SEARCH_DOCS
# - Schema: DATA
# - Stages: DOCS: it includes four corporate 10K Reports: Google, Meta, Microsoft, and Tesla
# - Generated Tables: DOCS_CHUNKS_TABLE

# ## Setup
# - Following this [website](https://quickstarts.snowflake.com/guide/getting_started_with_llmops_using_snowflake_cortex_and_trulens/index.html#1) for environment set up

# Once we have an environment with the right packages installed, we can load our credentials and set our Snowflake connection in a jupyter notebook notebook.
# %%
from dotenv import load_dotenv
from snowflake.snowpark import Session
import os

from glob import glob
from snowflake.connector import connect
from snowflake.cortex import Complete

load_dotenv(dotenv_path=".env")

# Load connection parameters from environment variables
import os
from snowflake.connector import connect

# Load connection parameters from environment variables
connection_params = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA")
}

# Ensure all required environment variables are set
for key, value in connection_params.items():
    if not value:
        raise ValueError(f"Environment variable '{key}' is not set.")

# %%
# Fetch the Cortex Search Service name
cortex_search_service_name = os.getenv("SNOWFLAKE_CORTEX_SEARCH_SERVICE")

# Ensure required environment variables are set
if not cortex_search_service_name:
    raise ValueError("Environment variable 'SNOWFLAKE_CORTEX_SEARCH_SERVICE' is not set.")

# Create a Snowpark session
snowpark_session = Session.builder.configs(connection_params).create()

# Print connection and service details for debugging
print("Connected to Snowflake:")
print(f"Account: {connection_params['account']}")
print(f"Database: {connection_params['database']}")
print(f"Schema: {connection_params['schema']}")
print(f"Cortex Search Service: {cortex_search_service_name}")

# %% [markdown]
# ## List stage documents for QC

# %%
#  Snowflake LIST command
list_command = "LIST @docs;"

try:
    # Establish the connection
    conn = connect(**connection_params)
    cursor = conn.cursor()
    
    # Execute the LIST command
    cursor.execute(list_command)
    
    # Fetch and print the results
    print("Files in stage '@docs':")
    for row in cursor:
        print(row)
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Ensure resources are cleaned up
    cursor.close()
    conn.close()


# %% [markdown]
# ## Cortex Search
# 
# Next, we'll turn to the retrieval component of our RAG and set up Cortex Search.
# 
# This requires three steps:
# 
# 1. Read and preprocess unstructured documents.
# 2. Embed the cleaned documents with Arctic Embed.
# 3. Call the Cortex search service.

# %% [markdown]
# ### Read and preprocess unstructured documents
# 
# For this example, we want to load Cortex Search with documentation from Github about a popular open-source library, Streamlit. To do so, we'll use a GitHub data loader available from LlamaHub.
# 
# Here we'll also expend some effort to clean up the text so we can get better search results.

# %% [markdown]
# ### Process the documents with Semantic Splitting
# 
# We'll use Snowflake's Arctic Embed model available from HuggingFace to embed the documents. We'll also use Llama-Index's `SemanticSplitterNodeParser` for processing.

# %%
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core.node_parser import SemanticSplitterNodeParser

# embed_model = HuggingFaceEmbedding("Snowflake/snowflake-arctic-embed-m")

# splitter = SemanticSplitterNodeParser(
#   buffer_size=1, breakpoint_percentile_threshold=85, embed_model=embed_model
# ) 

# %% [markdown]
# With the embed model and splitter, we can execute them in an ingestion pipeline

# %%
# from llama_index.core.ingestion import IngestionPipeline

# cortex_search_pipeline = IngestionPipeline(
#   transformations=[
#     splitter,
#   ],
# )

# results = cortex_search_pipeline.run(show_progress=True, documents=cleaned_documents)

# %% [markdown]
# ### Load data to Cortex Search
# 
# Now that we've embedded our documents, we're ready to load them to Cortex Search.
# 
# Here we can use the same connection details as we set up for Cortex Complete.

# %%
# SQL commands to set database and schema and fetch data
set_database = f"USE DATABASE {connection_params['database']};"
set_schema = f"USE SCHEMA {connection_params['schema']};"
sql_command = "SELECT * FROM docs_chunks_table LIMIT 10;"

try:
    # Establish the connection
    conn = connect(**connection_params)
    cursor = conn.cursor()
    
    # Set database and schema
    cursor.execute(set_database)
    cursor.execute(set_schema)
    
    # Execute the SQL command
    cursor.execute(sql_command)
    
    # Fetch results
    results = cursor.fetchall()
    for row in results:
        print(row)
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Ensure resources are cleaned up
    cursor.close()
    conn.close()

# %% [markdown]
# ### Call the Cortex Search Service
# 
# Next, we can go back to our python notebook and create a `CortexSearchRetreiver` class to connect to our cortex search service and add the `retrieve` method that we can leverage for calling it.

# %%
import os
from snowflake.core import Root
from typing import List

class CortexSearchRetriever:

    def __init__(self, session: Session, limit_to_retrieve: int = 4):
        self._session = session
        self._limit_to_retrieve = limit_to_retrieve

    def retrieve(self, query: str) -> List[str]:
        root = Root(self._session)
        cortex_search_service = (
        root
        .databases[os.environ["SNOWFLAKE_DATABASE"]]
        .schemas[os.environ["SNOWFLAKE_SCHEMA"]]
        .cortex_search_services[os.environ["SNOWFLAKE_CORTEX_SEARCH_SERVICE"]]
    )
        resp = cortex_search_service.search(
                query=query,
                columns=["chunk"],
                limit=self._limit_to_retrieve,
            )

        if resp.results:
            return [curr["chunk"] for curr in resp.results]
        else:
            return []

# %% [markdown]
# Once the retriever is created, we can test it out. Now that we have grounded access to the Streamlit docs, we can ask questions about using Streamlit, like "How do I launch a streamlit app".

# %%
## RAG example
        
from snowflake.cortex import Complete
query="who is ceo of microsoft?"
retriever = CortexSearchRetriever(snowpark_session, limit_to_retrieve=4)

retrieved_context = retriever.retrieve(query=query)
context = "\n".join(chunk for chunk in retrieved_context)
prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
print(f"Prompt:\n{prompt}")

print(Complete("mistral-large", prompt))

# %% [markdown]
# ## Create a RAG with built-in observability
# 
# Now that we've set up the components we need from Snowflake Cortex, we can build our RAG.
# 
# We'll do this by creating a custom python class with each the methods we need. We'll also add TruLens instrumentation with the `@instrument` decorator to our app.
# 
# The first thing we need to do however, is to set the database connection where we'll log the traces and evaluation results from our application. This way we have a stored record that we can use to understand the app's performance. This is done when initializing `Tru`.

# %%
from trulens.core import TruSession
from trulens.connectors.snowflake import SnowflakeConnector

tru_snowflake_connector = SnowflakeConnector(snowpark_session=snowpark_session)

tru_session = TruSession(connector=tru_snowflake_connector)

# %% [markdown]
# Now we can construct the RAG.

# %%
from trulens.apps.custom import instrument


class RAG_from_scratch:

    def __init__(self):
        self.retriever = CortexSearchRetriever(snowpark_session, limit_to_retrieve=4)

    @instrument
    def retrieve_context(self, query: str) -> list:
        """
        Retrieve relevant text from vector store.
        """
        return self.retriever.retrieve(query)

    @instrument
    def generate_completion(self, query: str, context_str: list) -> str:
        """
        Generate answer from context.
        """
        prompt = f"""
          You are an expert assistant extracting information from context provided.
          Answer the question based on the context. Be concise and do not hallucinate.
          If you don´t have the information just say so.
          Context: {context_str}
          Question:
          {query}
          Answer:
        """
        return Complete("mistral-large", prompt)

    @instrument
    def query(self, query: str) -> str:
        context_str = self.retrieve_context(query)
        return self.generate_completion(query, context_str)


rag = RAG_from_scratch()


# %% [markdown]
# After constructing the RAG, we can set up the feedback functions we want to use to evaluate the RAG.
# 
# Here, we'll use the [RAG Triad](https://www.trulens.org/trulens_eval/getting_started/core_concepts/rag_triad/). The RAG triad is made up of 3 evaluations along each edge of the RAG architecture: context relevance, groundedness and answer relevance.
# 
# Satisfactory evaluations on each provides us confidence that our LLM app is free from hallucination.
# 
# We will also use [LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) evaluations, using Mistral Large on [Snowflake Cortex](https://www.trulens.org/trulens_eval/api/provider/cortex/) as the LLM.

# %%
from trulens.providers.cortex.provider import Cortex
# from trulens.providers import Cortex
# from trulens_core import Cortex
# from trulens_connectors_snowflake import Cortex


from trulens.core import Feedback
from trulens.core import Select
import numpy as np

# provider = Cortex(snowpark_session.connection, "llama3.1-8b")
provider = Cortex(snowpark_session, "mistral-large")


f_groundedness = (
    Feedback(
    provider.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on(Select.RecordCalls.retrieve_context.rets[:].collect())
    .on_output()
)

f_context_relevance = (
    Feedback(
    provider.context_relevance,
    name="Context Relevance")
    .on_input()
    .on(Select.RecordCalls.retrieve_context.rets[:])
    .aggregate(np.mean)
)

f_answer_relevance = (
    Feedback(
    provider.relevance,
    name="Answer Relevance")
    .on_input()
    .on_output()
    .aggregate(np.mean)
)

feedbacks = [f_context_relevance,
            f_answer_relevance,
            f_groundedness,
        ]

# %% [markdown]
# After defining the feedback functions to use, we can just add them to the application along with giving the application an ID.

# %%
# from trulens_eval import TruCustomApp
# tru_rag = TruCustomApp(rag,
#     app_id = 'RAG v1',
#     feedbacks = [f_groundedness, f_answer_relevance, f_context_relevance])

# Ensure that the connection is within an active transaction

from trulens.apps.custom import TruCustomApp

tru_rag = TruCustomApp(
    rag,
    app_name="RAG",
    app_version="simple",
    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
    )


# %%
prompts = [
"What is Tesla's mission statement?",
"What vehicles does Tesla currently manufacture?",
"What is the function of the Tesla Powerwall?",
"How does Tesla's Supercharger network support renewable energy?",
"What are Tesla's two primary business segments?",
"What competitive advantage does Tesla have in energy storage systems?",
"What regulatory credits does Tesla earn and sell?",
"How does Tesla define its approach to full self-driving technology?",
"What is the purpose of Tesla’s Solar Roof product?",
"How does Tesla address supply chain risks?",
"What are Tesla's key strategies for lowering manufacturing costs?",
"What is Tesla's strategy for global manufacturing expansion?",
"What safety standards must Tesla vehicles comply with in the U.S.?",
"What is Tesla’s approach to customer-direct vehicle sales?",
"What are the primary risks facing Tesla as mentioned in its 10-K?",
"How does Tesla integrate sustainability into its operations?",
"What types of warranties does Tesla offer for its energy products?",
"What financial options does Tesla offer for its solar customers?",
"How does Tesla mitigate risks associated with battery transport?",
"What ESG initiatives are highlighted in Tesla’s report?"
]

# %% [markdown]
# Now that the application is ready, we can run it on a test set of questions about streamlit to measure its performance.

# %%
# with tru_rag as recording:
#     for prompt in prompts:
#         rag.query(prompt)

# tru_session.get_leaderboard()

with tru_rag as recording:
    for prompt in prompts:
        try:
            rag.query(prompt)
        except Exception as e:
            print(f"Error processing prompt '{prompt}': {e}")
            # Optionally log the error or take other actions here

# Retrieve the leaderboard even if some queries fail
try:
    tru_session.get_leaderboard()
except Exception as e:
    print(f"Error retrieving leaderboard: {e}")


# %%
tru_session.get_leaderboard()

# %% [markdown]
# ## Use Guardrails
# 
# In addition to making informed iteration, we can also directly use feedback results as guardrails at inference time. In particular, here we show how to use the context relevance score as a guardrail to filter out irrelevant context before it gets passed to the LLM. This both reduces hallucination and improves efficiency.
# 
# To do so, we'll rebuild our RAG using the `@context-filter` decorator on the method we want to filter, and pass in the feedback function and threshold to use for guardrailing.

# %%
from trulens.core.guardrails.base import context_filter

# note: feedback function used for guardrail must only return a score, not also reasons
f_context_relevance_score = Feedback(
    provider.context_relevance, name="Context Relevance"
)


class filtered_RAG_from_scratch(RAG_from_scratch):

    @instrument
    @context_filter(f_context_relevance_score, 0.75, keyword_for_prompt="query")
    def retrieve_context(self, query: str) -> list:
        """
        Retrieve relevant text from vector store.
        """
        return self.retriever.retrieve(query)


filtered_rag = filtered_RAG_from_scratch()

# %% [markdown]
# We can combine the new version of our app with the feedback functions we already defined

# %%
from trulens.apps.custom import TruCustomApp

tru_filtered_rag = TruCustomApp(
    filtered_rag,
    app_name="RAG",
    app_version="filtered",
    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
)

# %% [markdown]
# Then we run it on a test set of questions about tesla 10K to measure its performance.

# %%
# Main loop with error handling
with tru_filtered_rag as recording:
    for prompt in prompts:
        try:
            filtered_rag.query(prompt)
        except Exception as e:
            print(f"Error processing prompt '{prompt}': {e}")

# Retrieve the leaderboard even if some queries fail
try:
    tru_session.get_leaderboard()
except Exception as e:
    print(f"Error retrieving leaderboard: {e}")

# %%
tru_session.get_leaderboard()

# %%

filtered_leaderboard=tru_session.get_leaderboard()
filtered_leaderboard
# %%
import pandas as pd

# Simulate leaderboard data (replace with actual retrievals)
rag_leaderboard = tru_session.get_leaderboard("RAG")
filtered_leaderboard = tru_session.get_leaderboard("RAG Filtered")

# Convert to DataFrame for comparison
df_rag = pd.DataFrame(rag_leaderboard)
df_filtered = pd.DataFrame(filtered_leaderboard)

# Compute differences
comparison = pd.concat([df_rag.mean(), df_filtered.mean()], axis=1)
comparison.columns = ["RAG", "Filtered RAG"]

print("Comparison of Leaderboard Metrics:")
print(comparison)
# %%

# ## Conclusion And Resources
# 
# Congratulations! You've successfully built a RAG by combining Cortex Search and LLM Functions, adding in TruLens Feedback Functions as Observability. You also set up logging for TruLens to Snowflake, and added TruLens Guardrails to reduce hallucination.
# 
# ### What You Learned
# 
# - How to build a RAG with Cortex Search and Cortex LLM Functions.
# - How to use TruLens Feedback Functions and Tracing.
# - How to log TruLens Evaluation Results and Traces to Snowflake.
# - How to use TruLens Feedback Functions as Guardrails to reduce hallucination.
# 
# ### Related Resources
# 
# - [Snowflake Cortex Documentation](https://docs.snowflake.com/en/guides-overview-ai-features)
# - [TruLens Documentation](https://trulens.org/)
# - [TruLens GitHub Repository](https://github.com/truera/trulens)

# %%
import subprocess

# Launch the Streamlit app in a non-blocking subprocess
process = subprocess.Popen(["streamlit", "run", "app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

print("Streamlit app is running. Open http://localhost:8501 in your browser.")

# Note: Do not call process.communicate() here, as it will block indefinitely.


# %%



