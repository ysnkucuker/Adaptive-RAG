from typing import List, TypedDict

from langchain.chains.hyde.prompts import web_search


class GraphState(TypedDict):
    """
    Represent the state of a graph
    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search or not
        documents: list of documents
    """

question: str
generation: str
web_search: bool
documents: List[str]
