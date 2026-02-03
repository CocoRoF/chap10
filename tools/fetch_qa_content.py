# GitHub: https://github.com/naotaka1128/llm_app_codes/chapter_010/tools/fetch_qa_content.py

from pathlib import Path
import streamlit as st
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field

# ============================================================
# 크로스 플랫폼 경로 처리 (Windows / macOS / Linux 호환)
# ============================================================
# 상대 경로(예: "./vectorstore")는 현재 작업 디렉토리(CWD) 기준으로 해석되어
# 실행 위치에 따라 FileNotFoundError가 발생할 수 있음.
# pathlib.Path와 __file__을 사용하여 스크립트 파일 위치 기준의 절대 경로를 계산함으로써
# 어떤 운영체제에서 어떤 디렉토리에서 실행하더라도 올바른 경로를 참조하도록 함.
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent


class FetchQAContentInput(BaseModel):
    """타입을 지정하기 위한 클래스"""

    query: str = Field()


# ============================================================
# Streamlit 캐시 설정 (show_spinner=False)
# ============================================================
# LangGraph 에이전트는 별도의 스레드(ThreadPoolExecutor)에서 tool을 실행함.
# 이 스레드에는 Streamlit 세션 컨텍스트가 없어서 st.cache_resource의
# 기본 spinner가 NoSessionContext 에러를 발생시킴.
# show_spinner=False로 설정하여 이 문제를 해결함.
# ============================================================
@st.cache_resource(show_spinner=False)
def load_qa_vectorstore():
    """'자주 묻는 질문' 벡터 DB를 로드"""
    vectorstore_path = BASE_DIR / "vectorstore" / "qa_vectorstore"
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(
        str(vectorstore_path), embeddings=embeddings, allow_dangerous_deserialization=True
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
