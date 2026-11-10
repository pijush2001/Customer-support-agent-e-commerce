import os
import ssl
from contextlib import asynccontextmanager
from urllib import response
import httpx
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from langserve import add_routes
load_dotenv()
groq_api_key = os.getenv("Groq_Api_Key")

    

    
def create_app() -> FastAPI:
    ssl_context = ssl.create_default_context()
    client = httpx.Client(verify=ssl_context)
    async_client = httpx.AsyncClient(verify=ssl_context)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            await async_client.aclose()
            client.close()

    model = ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=groq_api_key,
        http_client=client,
        http_async_client=async_client,
    )
    # messages =[
    #     SystemMessage(content="Translate the following from English to bengali"),
    #     HumanMessage(content="Hello, how are you?")
    # ]
    ##response = model.invoke(messages)
    parser = StrOutputParser()
    #parsed_response = parser.invoke(response)
    
    ### using LCEL - Chain the component
    ##chain = model |parser
    ##chained_response = chain.invoke(messages)
    ## prompt template
    generic_template = "Translate the following into {language}: "
    prompt = ChatPromptTemplate.from_messages(
        [("system", generic_template),("user", "{user_text}")]
    )
    #prompt_response = prompt.invoke({"language": "bengali", "user_text": "Good Morning"})
    chain = prompt|model|parser
    ##App definition
    app = FastAPI(
        title="Learning Agent",
        version="0.1.0",
        description="A FastAPI app for the Learning Agent",
        lifespan=lifespan,
    )
    ##adding chain route
    add_routes(
        app,
        chain,
        path ="/chain"
    )
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)