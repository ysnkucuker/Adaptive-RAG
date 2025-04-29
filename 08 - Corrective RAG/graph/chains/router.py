from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

class RouteQuery(BaseModel):
    """
        Route a suer query to the most relevant datasource
    """
    datasource : Literal["vectorstore", "websearch"] = Field(
        ..., # Burası ya vector ya web
        description="Given a user question choose to route it to web search or a vectorstore"
    )

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """
You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documnets realted to agents, prompt engineering and adverserial attacks on llms.
Use the vectorstore for questions on the topics. For all else, use web-search.
"""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}")
    ]
)

question_route = route_prompt | structured_llm_router


#if __name__ == "__main__":
#    print(question_route.invoke(
#        {"question": "what is ai agent"}
#    ))
