import streamlit as st
from snowflake.snowpark.context import get_active_session
from snowflake.cortex import Complete
from snowflake.core import Root
import pandas as pd
import json

pd.set_option("max_colwidth", None)

# Default Values
NUM_CHUNKS = 3
slide_window = 7

# Service Parameters
CORTEX_SEARCH_DATABASE = "FINANCE_CORTEX_SEARCH_DOCS"
CORTEX_SEARCH_SCHEMA = "DATA"
CORTEX_SEARCH_SERVICE = "FinancialBot_SEARCH_SERVICE_CS"

# Columns to Query in the Service
COLUMNS = ["chunk", "relative_path", "category"]

session = get_active_session()
root = Root(session)
svc = root.databases[CORTEX_SEARCH_DATABASE].schemas[CORTEX_SEARCH_SCHEMA].cortex_search_services[CORTEX_SEARCH_SERVICE]

# Functions

def config_options():
    st.sidebar.selectbox('Select your model:', (
        'mixtral-8x7b', 'snowflake-arctic', 'mistral-large', 'llama3-8b',
        'llama3-70b', 'reka-flash', 'mistral-7b', 'llama2-70b-chat', 'gemma-7b'
    ), key="model_name")

    categories = session.table('DOCS_CHUNKS_TABLE').select('category').distinct().collect()
    cat_list = ['ALL'] + [cat.CATEGORY for cat in categories]
    st.sidebar.selectbox('Select product category:', cat_list, key="category_value")

    st.sidebar.checkbox('Remember chat history?', key="use_chat_history", value=True)
    st.sidebar.checkbox('Debug: Show previous conversation summary', key="debug", value=True)
    st.sidebar.button("Start Over", key="clear_conversation", on_click=init_messages)
    st.sidebar.expander("Session State").write(st.session_state)

def init_messages():
    if st.session_state.clear_conversation or "messages" not in st.session_state:
        st.session_state.messages = []

def get_similar_chunks_search_service(query):
    filter_obj = ("ALL" if st.session_state.category_value == "ALL" else {"@eq": {"category": st.session_state.category_value}})
    response = svc.search(query, COLUMNS, filter=(filter_obj if filter_obj != "ALL" else None), limit=NUM_CHUNKS)
    st.sidebar.json(response.json())
    return response.json()

def get_chat_history():
    start_index = max(0, len(st.session_state.messages) - slide_window)
    return st.session_state.messages[start_index:-1]

def summarize_question_with_history(chat_history, question):
    prompt = f"""
        Based on the chat history below and the question, generate a query that extends the question
        with the provided chat history. Respond with only the query.
        <chat_history>{chat_history}</chat_history>
        <question>{question}</question>
    """
    summary = Complete(st.session_state.model_name, prompt)
    if st.session_state.debug:
        st.sidebar.text("Summary for context:")
        st.sidebar.caption(summary)
    return summary.replace("'", "")

def create_prompt(myquestion):
    chat_history = get_chat_history() if st.session_state.use_chat_history else []
    if chat_history:
        question_summary = summarize_question_with_history(chat_history, myquestion)
        prompt_context = get_similar_chunks_search_service(question_summary)
    else:
        prompt_context = get_similar_chunks_search_service(myquestion)

    prompt = f"""
        You are an expert assistant extracting information from the CONTEXT.
        Use the CHAT HISTORY if provided. Answer the QUESTION concisely, and if information
        is unavailable, state so. Do not reference CONTEXT or CHAT HISTORY in your answer.
        <chat_history>{chat_history}</chat_history>
        <context>{prompt_context}</context>
        <question>{myquestion}</question>
    """
    relative_paths = set(item['relative_path'] for item in json.loads(prompt_context)['results'])
    return prompt, relative_paths

def answer_question(myquestion):
    prompt, relative_paths = create_prompt(myquestion)
    response = Complete(st.session_state.model_name, prompt)
    return response, relative_paths

def main():
    st.title(":speech_balloon: Chat with Tech 10K report with Snowflake Cortex")

    # Document Listing
    st.write("Available documents for answering queries:")
    docs_available = session.sql("ls @docs").collect()
    document_names = [doc["name"] for doc in docs_available]
    st.dataframe(pd.DataFrame(document_names, columns=["Document Names"]))

    config_options()
    init_messages()

    # Chat Messages Display
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept User Input
    if question := st.chat_input("What do you want to know about your products?"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner(f"{st.session_state.model_name} thinking..."):
                response, relative_paths = answer_question(question)
                message_placeholder.markdown(response)
                if relative_paths:
                    with st.sidebar.expander("Related Documents"):
                        for path in relative_paths:
                            cmd2 = f"SELECT GET_PRESIGNED_URL(@docs, '{path}', 360) AS URL_LINK FROM DIRECTORY(@docs)"
                            url_link = session.sql(cmd2).to_pandas().iloc[0]['URL_LINK']
                            st.sidebar.markdown(f"Doc: [{path}]({url_link})")
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

