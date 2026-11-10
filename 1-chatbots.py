import os
import ssl
import httpx
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("Groq_Api_Key")
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

with httpx.Client(verify=ssl.create_default_context()) as client:
    model = ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=groq_api_key,
        http_client=client,
    )
    response = model.invoke([HumanMessage(content="Hello, My name is Pijush and I am a dotnet developer")])
    response2 = model.invoke([
        HumanMessage(content="What is my name?"),
        AIMessage(content=response.content),
        HumanMessage(content="Please remind me what type of skill do I have?")
    ])
    ## message history
    from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory
    store={}
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]
    config = {"configurable":{"session_id": "default_session"}}
    with_message_history = RunnableWithMessageHistory(
        model,
        get_session_history
    )
    response =with_message_history.invoke([HumanMessage(content="Hi, my name is Pijush and I am a dotnet developer")],
                                config=config)
    print(response.content)