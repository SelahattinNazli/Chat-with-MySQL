import streamlit as st

# Streamlit app configuration
st.set_page_config(page_title="Chat with MySQL Database", page_icon=":speech_balloon:")

from dotenv import load_dotenv
import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Load environment variables from a .env file
load_dotenv()

# Ensure chat history is initialized in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]

# Ensure db is initialized in session state
if 'db' not in st.session_state:
    st.session_state.db = None

# Access the OpenAI API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable in your .env file.")
else:
    st.write(f"OpenAI API Key Loaded: {openai_api_key[:5]}...")  # Only print the first few characters for security

def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: which user likes the most photos?
    SQL Query: SELECT User_id, COUNT(Photo_id) as track_count FROM Likes GROUP BY user_id ORDER BY user_id ASC LIMIT 1
    Question: Name 2 TAGS
    SQL Query: SELECT Tag_Name FROM Tags LIMIT 2;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    
    def get_schema(_):
        return db.get_table_info()
    
    return RunnableSequence(
        RunnablePassthrough.assign(schema=get_schema),
        prompt,
        llm,
        StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

# App title
st.title("Chat with MySQL")

# Sidebar for database settings
with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application using MySQL. Connect to the database and start chatting.")
    
    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Port", value="3306", key="Port")
    st.text_input("User", value="root", key="User")
    st.text_input("Password", type="password", value="Sn010658?", key="Password")
    st.text_input("Database", value="ig_clone", key="Database")
    
    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db = init_database(
                st.session_state["User"],
                st.session_state["Password"],
                st.session_state["Host"],
                st.session_state["Port"],
                st.session_state["Database"]
            )
            st.session_state.db = db
            st.success("Connected to database!")

# Display chat history
if 'chat_history' in st.session_state:
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)

# Add new user query to chat history
user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI"):
        if st.session_state.db is not None:
            response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        else:
            response = "Database not connected."
        st.markdown(response)
        
    st.session_state.chat_history.append(AIMessage(content=response))
    st.rerun()  # Replace st.experimental_rerun with st.rerun