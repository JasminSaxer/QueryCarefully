import streamlit as st
from src.final_pipeline import QueryCarefullyPipeline 
from src.few_shot_nearest.few_shot_nearest import question_embedding
import logging
import time

def stream_data(data):
    for word in data.split(" "):
        yield word + " "
        time.sleep(0.02)
        
# Suppress Streamlit and other library debug logs
logging.getLogger("watchdog").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

# Set Streamlit page config to use full width
st.set_page_config(layout="wide")

# get embedding model ready (faster processing when loaded once)
# embedding_model = question_embedding('Alibaba-NLP/gte-Qwen2-1.5B-instruct')
embedding_model = None

## main application
st.title('Query Carefully - User Interface')
st.write('This is a simple user interface for the Query Carefully pipeline. You can enter your question for the OncoMX database and get the result from the database.')


# Add a dropdown to select the LLM model
llm_model = st.selectbox(
    "Choose LLM model",
    options=[
        "llama3.3:70b", 
        'mistral-small:24b'
    ],
    index=0
)
st.session_state["llm_model"] = llm_model


if "messages" not in st.session_state:
    st.session_state.messages = []


with st.chat_message("assistant"):
    start_prompt_assistant = "Hello human, please enter your question for the OncoMX database. I will try to answer it as best as I can."
    st.write_stream(stream_data(start_prompt_assistant))
    st.session_state.messages.append({"role": "assistant", "content": start_prompt_assistant})


# if you have a chat input
if prompt := st.chat_input("Ask away!"):

    with st.chat_message("user"):
        st.markdown(prompt)
        # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    try:
        # run pipeline to get result
        with st.status("Processing your question...", expanded=True) as status:
            res_type, result = QueryCarefullyPipeline(prompt, st, embedding_model=embedding_model)
            status.update(
                label="Finished!", state="complete", expanded=False
            )
                        
        # if return rephrase, option to rephrase the question..
        if res_type == 'rephrase':
            with st.chat_message("assistant"):
                st.markdown(result)
            
        # if result show the result
        if res_type == 'result':
            sql_query, LM_response, db_res, res_explanation = result
                        
            if sql_query is None:
                with st.chat_message("assistant"):
                    no_result_answer = "I could not generate a SQL query for your question. Please try again."
                    st.markdown(no_result_answer)
                st.session_state.messages.append({"role": "assistant", "content": no_result_answer})
            else:
                with st.chat_message("assistant"):
                    st.markdown("Here is the SQL query I generated:")
                    st.code(sql_query)
                    
                    st.write_stream(stream_data("Here is the response from the database:"))
                    st.write(db_res)

                    st.write_stream(stream_data(res_explanation))
                
                st.session_state.messages.append({"role": "assistant", "content": "Here is the SQL query I generated:"})
                st.session_state.messages.append({"role": "assistant", "content": sql_query})
                st.session_state.messages.append({"role": "assistant", "content": "Here is the response from the database:"})
                st.session_state.messages.append({"role": "assistant", "content": db_res})
                st.session_state.messages.append({"role": "assistant", "content": res_explanation})

        if res_type == 'error':
            with st.chat_message("assistant"):
                st.write_stream(stream_data("I could not process your question. Please try again."))
            st.session_state.messages.append({"role": "assistant", "content": "I could not process your question. Please try again."})

    
    except Exception as e:
        with st.chat_message("assistant"):
            st.write_stream(stream_data("An error occurred while processing your question. Please try again."))
            st.write_stream(stream_data(f"Error: {e}"))
            st.stop()
        
