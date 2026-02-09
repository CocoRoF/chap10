import streamlit as st
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field

class FetchQAContentInput(BaseModel):
    """타입을 지정하기 위한 클래스"""

    query: str = Field()

@st.cache_resource(show_spinner=False)
def load_qa_vectorstore(vectorstore_path="./vectorstore/qa_vectorstore"):
    """'자주 묻는 질문' 벡터 DB를 로드"""
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(
        vectorstore_path, embeddings=embeddings, allow_dangerous_deserialization=True
    )

@tool(args_schema=FetchQAContentInput)
def fetch_qa_content(query):
    """
    '자주 묻는 질문' 리스트 중에서, 사용자의 질문과 관련된 콘텐츠를 찾아주는 도구입니다.
    '영진모바일'에 관한 구체적인 정보를 얻는 데 도움이 됩니다.

    이 도구는 `similarity`(유사도)와 `content`(콘텐츠)를 반환합니다.
    - 'similarity'는 답변이 질문과 얼마나 관련되어 있는지를 나타냅니다.
        값이 높을수록 질문과의 관련성이 높다는 의미입니다.
        'similarity' 값이 0.5 미만인 문서는 반환되지 않습니다.
    - 'content'는 질문에 대한 답변 텍스트를 제공합니다.
        일반적으로 자주 묻는 질문과 그에 대응하는 답변으로 구성됩니다.

    빈 리스트가 반환된 경우, 사용자의 질문에 대한 답변을 찾지 못했다는 의미입니다.
    그런 경우 질문 내용을 좀 더 명확히 요청하는 것이 좋습니다.

    Returns
    -------
    List[Dict[str, Any]]:
    - page_content
      - similarity: float
      - content: str
    """
    db = load_qa_vectorstore()
    docs = db.similarity_search_with_score(query=query, k=5, score_threshold=0.5)
    return [
        {"similarity": 1 - similarity, "content": i.page_content}
        for i, similarity in docs
    ]
