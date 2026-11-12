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
    # direct_chat_response = model.invoke([HumanMessage(content="Hello, My name is Pijush and I am a dotnet developer")])
    # response_from_previous_info = model.invoke([
    #     HumanMessage(content="What is my name?"),
    #     AIMessage(content=direct_chat_response.content),
    #     HumanMessage(content="Please remind me what type of skill do I have?")
    # ])
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
    response_with_history =with_message_history.invoke([HumanMessage(content="Hi, my name is Pijush and I am a dotnet developer")],
                                config=config)
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant. Answer all the question to the best of your ability in {language}",),
         MessagesPlaceholder(variable_name="messages")]
    )
    chain = prompt|model

    #response = chain.invoke({"messages": [HumanMessage(content="Hi, my name is Pijush and I am a dotnet developer")], "language": "Bengali"})
    #print(response.content)
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="messages"
    )
    config = {"configurable":{"session_id": "chat4"}}
    #response_with_prompt_history = with_message_history.invoke({"messages": [HumanMessage(content="I am Pijush, dot net developer having 4.5 years of experience"), HumanMessage(content="Can you summarize my profile?")], "language": "Bengali"},
    #                            config=config)
    # ##print(response.content)
    #print(response_with_prompt_history.content)
    from langchain_core.messages import SystemMessage, trim_messages
    session_history = get_session_history("chat4")
    
    messages1 = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hi, my name is Pijush"),
        AIMessage(content="Hello Pijush, nice to meet you."),
        HumanMessage(content="I love khichdi"),
        AIMessage(content="Khichdi is delicious!"),
        HumanMessage(content="Can you suggest a recipe for khichdi?"),
        AIMessage(content="Sure! Here's a simple khichdi recipe..."),
        HumanMessage(content="Thank you! That looks easy to make."),
        AIMessage(content="You're welcome! Enjoy your khichdi.")
    ]
    from langchain_core.messages import trim_messages
    trimmer = trim_messages(
        max_tokens=100,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=True,
        start_on="human"
    )
   
    #print(trimmed_messages)
    from operator import itemgetter
    from langchain_core.runnables import RunnablePassthrough

    chain = (
        RunnablePassthrough.assign(messages = itemgetter("messages") | trimmer) | prompt | model
    )
    response_Trim = chain.invoke({"messages": messages1 + [HumanMessage(content="What do I love to eat?")],
                                   "language": "english"})
    ##print(response_Trim.content)
    config = {"configurable":{"session_id": "chat4"}}
    response_with_history = with_message_history.invoke({"messages": messages1 + [HumanMessage(content="What do I love to eat?")],
                                                        "language": "english"},
                                                       config=config)
    print(response_with_history.content)