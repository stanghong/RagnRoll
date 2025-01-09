import streamlit as st

# -- For legacy Streamlit rerun hack --
try:
    from streamlit.scriptrunner import RerunException
    from streamlit.scriptrunner.script_run_context import ScriptRunContext
except ImportError:
    RerunException = None
    ScriptRunContext = None

def custom_rerun():
    """
    For Streamlit < 1.10:
    Manually raise RerunException to restart the script.
    If these classes can't be imported, it won't work and
    the user should upgrade or verify their Streamlit version.
    """
    if RerunException and ScriptRunContext:
        raise RerunException(ScriptRunContext())
    else:
        st.warning(
            "Your Streamlit version might not support the custom rerun hack. "
            "Please upgrade to Streamlit >= 1.10 or verify imports."
        )

# ------------------------------------------------------------------------
# Original code begins below
# ------------------------------------------------------------------------

from snowflake.snowpark.context import get_active_session
from snowflake.cortex import Complete
from snowflake.core import Root
import pandas as pd
import json

pd.set_option("max_colwidth", None)

# -- Default Values
NUM_CHUNKS = 4  # Starting chunk limit
MAX_CHUNKS = 8  # Maximum chunk limit
SLIDE_WINDOW = 7

# -- Service Parameters
CORTEX_SEARCH_DATABASE = "FINANCE_CORTEX_SEARCH_DOCS"
CORTEX_SEARCH_SCHEMA = "DATA"
CORTEX_SEARCH_SERVICE = "FinancialBot_SEARCH_SERVICE_CS"

# -- Columns to Query in the Service
COLUMNS = ["chunk", "relative_path", "category"]

session = get_active_session()
root = Root(session)
svc = root.databases[CORTEX_SEARCH_DATABASE].schemas[CORTEX_SEARCH_SCHEMA].cortex_search_services[CORTEX_SEARCH_SERVICE]

def notion_integration():
    """
    Placeholder function to handle Notion integration (e.g., send feedback, log data).
    """
    st.success("Notion integration triggered! (placeholder)")

def config_options():
    st.sidebar.selectbox('Select your model:', (
        'mixtral-8x7b', 'snowflake-arctic', 'mistral-large', 'llama3-8b',
        'llama3-70b', 'reka-flash', 'mistral-7b', 'llama2-70b-chat', 'gemma-7b'
    ), key="model_name")

    # Example for category filter
    categories = session.table('DOCS_CHUNKS_TABLE').select('category').distinct().collect()
    cat_list = ['ALL'] + [cat.CATEGORY for cat in categories]
    st.sidebar.selectbox('Select corporate:', cat_list, key="category_value")

    st.sidebar.checkbox('Remember chat history?', key="use_chat_history", value=True)
    st.sidebar.checkbox('Debug: Show previous conversation summary', key="debug", value=True)
    st.sidebar.button("Start Over", key="clear_conversation", on_click=init_messages)
    st.sidebar.expander("Session State").write(st.session_state)

def init_messages():
    """
    Initialize or clear the stored conversation messages.
    """
    if st.session_state.clear_conversation or "messages" not in st.session_state:
        st.session_state.messages = []

def get_similar_chunks_search_service(query):
    """
    Uses the search service to retrieve similar chunks based on the query.
    The chunk limit is taken from st.session_state.num_chunks if it exists.
    """
    chunk_limit = st.session_state.get("num_chunks", NUM_CHUNKS)
    filter_obj = (
        "ALL"
        if st.session_state.category_value == "ALL"
        else {"@eq": {"category": st.session_state.category_value}}
    )

    response = svc.search(
        query,
        ["chunk", "relative_path", "category"],
        filter=(filter_obj if filter_obj != "ALL" else None),
        limit=chunk_limit
    )
    st.sidebar.json(response.json())
    return response.json()

def get_chat_history():
    """
    Returns the most recent conversation limited by 'slide_window'.
    """
    start_index = max(0, len(st.session_state.messages) - SLIDE_WINDOW)
    return st.session_state.messages[start_index:-1]

def summarize_question_with_history(chat_history, question):
    """
    Summarizes the user question along with the chat history
    to form a single query for better context.
    """
    prompt = f"""
        Based on the chat history below and the question, generate a query that extends the question
        with the provided chat history. Respond with only the query.
        <chat_history>{chat_history}</chat_history>
        <question>{question}</question>
    """
    summary = Complete(st.session_state.model_name, prompt)
    return summary.replace("'", "")

def create_prompt(myquestion):
    """
    Creates the final prompt for the LLM, including instructions to provide Answer Relevancy.
    """
    chat_history = get_chat_history() if st.session_state.use_chat_history else []

    if chat_history:
        question_summary = summarize_question_with_history(chat_history, myquestion)
        prompt_context = get_similar_chunks_search_service(question_summary)
    else:
        prompt_context = get_similar_chunks_search_service(myquestion)

    prompt = f"""
        You are an expert assistant extracting information from the CONTEXT.
        Use the CHAT HISTORY if provided. Answer the QUESTION concisely, and if information
        is unavailable, state so.
        Provide your evaluation of answer relevancy at the end in a separate line,
        the relevancy number is between 0-100% with 100% being high confidence and highly relevant, 0% is vice versa.

        Do not reference CONTEXT or CHAT HISTORY in your answer.

        <chat_history>{chat_history}</chat_history>
        <context>{prompt_context}</context>
        <question>{myquestion}</question>
    """

    # Collect relative paths from JSON
    relative_paths = set()
    try:
        results_json = json.loads(prompt_context)
        for item in results_json.get("results", []):
            relative_paths.add(item["relative_path"])
    except Exception:
        pass

    return prompt, relative_paths

def answer_question(myquestion):
    """
    Generates an LLM response using create_prompt and the chosen model.
    """
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

    # Ensure necessary session_state variables exist
    if "num_chunks" not in st.session_state:
        st.session_state.num_chunks = NUM_CHUNKS
    if "looping" not in st.session_state:
        st.session_state.looping = False
    if "current_question" not in st.session_state:
        st.session_state.current_question = ""

    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Step 1: Check if user entered a new query
    new_query = st.chat_input("What do you want to know about corporate 10K?")
    if new_query:
        # This is a brand-new question, so exit any thumbs-down loop
        st.session_state.looping = False  
        st.session_state.num_chunks = NUM_CHUNKS  # reset if you wish

        st.session_state.current_question = new_query
        st.session_state.messages.append({"role": "user", "content": new_query})

        # Generate the initial answer
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner(f"{st.session_state.model_name} thinking..."):
                response, relative_paths = answer_question(new_query)
                message_placeholder.markdown(response)

                if relative_paths:
                    with st.sidebar.expander("Related Documents"):
                        for path in relative_paths:
                            cmd2 = f"SELECT GET_PRESIGNED_URL(@docs, '{path}', 360) AS URL_LINK FROM DIRECTORY(@docs)"
                            url_link = session.sql(cmd2).to_pandas().iloc[0]['URL_LINK']
                            st.sidebar.markdown(f"Doc: [{path}]({url_link})")

            st.session_state.messages.append({"role": "assistant", "content": response})

        # Turn off the looping until user hits thumbs-down
        st.session_state.looping = False

    # Step 2: If there's a current_question in session_state, show thumbs-up/down
    if st.session_state.current_question:
        col1, col2 = st.columns(2)

        # Thumbs Up
        if col1.button("👍", key=f"thumbs_up_{len(st.session_state.messages)}"):
            # End loop
            st.session_state.looping = False
            notion_integration()
            # Re-run using the custom rerun function
            custom_rerun()

        # Thumbs Down
        if col2.button("👎", key=f"thumbs_down_{len(st.session_state.messages)}"):
            # Increase chunk limit if possible
            old_limit = st.session_state.num_chunks
            if old_limit < MAX_CHUNKS:
                st.session_state.num_chunks = min(old_limit + 4, MAX_CHUNKS)
                st.warning(
                    f"Chunk limit changed from {old_limit} to {st.session_state.num_chunks}. "
                    "Repeating the same query for more context..."
                )

                # Update question with extra instructions
                appended_q = (
                    f"{st.session_state.current_question} "
                    "add additional information and think step by step"
                )
                st.session_state.current_question = appended_q

                # Record the user "question" in messages for clarity
                st.session_state.messages.append({"role": "user", "content": appended_q})

                # Now set looping to True so we'll keep re-running until thumbs-up or new query
                st.session_state.looping = True
                # Re-run using the custom rerun function
                custom_rerun()
            else:
                st.info("Chunk limit is already at 8. No further increase possible.")
                st.session_state.looping = False

    # Step 3: The "looping" logic - keep re-generating answers until thumbs up / chunk limit = 8 / new query
    if st.session_state.looping and st.session_state.current_question:
        # Auto-generate a new answer with the updated chunk limit and question
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Re-running query with updated context..."):
                new_response, new_relative_paths = answer_question(st.session_state.current_question)
                message_placeholder.markdown(new_response)

                # Show updated docs
                if new_relative_paths:
                    with st.sidebar.expander("Related Documents (updated)"):
                        for path in new_relative_paths:
                            cmd2 = (
                                f"SELECT GET_PRESIGNED_URL(@docs, '{path}', 360) AS URL_LINK "
                                f"FROM DIRECTORY(@docs)"
                            )
                            url_link = session.sql(cmd2).to_pandas().iloc[0]['URL_LINK']
                            st.sidebar.markdown(f"Doc: [{path}]({url_link})")

            # Add new response to conversation
            st.session_state.messages.append({"role": "assistant", "content": new_response})

        # If chunk limit has reached 8, stop looping automatically
        if st.session_state.num_chunks >= MAX_CHUNKS:
            st.session_state.looping = False

        # The user can click thumbs-down again, or thumbs-up, or input a new query
        # which will set st.session_state.looping = False in the relevant logic above

if __name__ == "__main__":
    main()
