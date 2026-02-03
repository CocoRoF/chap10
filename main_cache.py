# GitHub: https://github.com/naotaka1128/llm_app_codes/chapter_010/main_cache_new.py

import streamlit as st
import uuid  # thread_id 생성용

# ============================================================
# [수정] LangChain 1.0.0+ 버전 대응
# - 기존: create_tool_calling_agent + AgentExecutor 조합
# - 변경: create_agent 단일 API로 통합 (더 간결한 코드)
# - 이유: LangChain 1.0.0에서 에이전트 생성 API가 단순화됨
# ============================================================
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware  # 대화 요약 미들웨어
from langgraph.checkpoint.memory import InMemorySaver  # 대화 상태 저장소

# models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# custom tools
from tools.fetch_qa_content import fetch_qa_content
from tools.fetch_stores_by_prefecture import fetch_stores_by_prefecture
from src.cache import Cache

# ============================================================
# [수정] 스트리밍 핸들러 변경
# - 기존: StreamlitCallbackHandler (LangChain 레거시)
# - 변경: StreamlitLanggraphHandler (LangGraph 호환)
# - 이유: create_agent가 내부적으로 LangGraph 기반으로 동작하므로
#         LangGraph 전용 핸들러 사용 필요
# ============================================================
from youngjin_langchain_tools import StreamlitLanggraphHandler

###### dotenv를 사용하지 않는 경우는 삭제해주세요 ######
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn(
        "dotenv not found. Please make sure to set your environment variables manually.",
        ImportWarning,
    )
################################################


@st.cache_data
def load_system_prompt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# ============================================================
# Streamlit UI Functions
# ============================================================
def init_page():
    st.set_page_config(page_title="고객 지원", page_icon="🐻")
    st.header("고객 지원🐻")
    st.sidebar.title("Options")


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        welcome_message = (
            "영진모바일 고객지원에 오신 것을 환영합니다. 질문을 입력해 주세요 🐻"
        )
        st.session_state.messages = [{"role": "assistant", "content": welcome_message}]
        # [수정] 메모리 관리 방식 변경
        # - 기존: ConversationBufferWindowMemory (LangChain 레거시)
        # - 변경: InMemorySaver + thread_id 조합 (LangGraph 방식)
        # - 이유: create_agent는 LangGraph 기반으로 동작하며, checkpointer를 통해 대화 상태를 관리함
        st.session_state["checkpointer"] = InMemorySaver()
        st.session_state["thread_id"] = str(uuid.uuid4())

    if len(st.session_state.messages) == 1:
        st.session_state["first_question"] = True
    else:
        st.session_state["first_question"] = False


def select_model(temperature=0):
    models = ("GPT-5 mini", "GPT-5.2", "Claude Sonnet 4.5", "Gemini 2.5 Flash")
    model = st.sidebar.radio("사용할 모델 선택:", models)
    if model == "GPT-5 mini":
        return ChatOpenAI(temperature=temperature, model="gpt-5-mini")
    elif model == "GPT-5.2":
        return ChatOpenAI(temperature=temperature, model="gpt-5.2")
    elif model == "Claude Sonnet 4.5":
        return ChatAnthropic(
            temperature=temperature, model="claude-sonnet-4-5-20250929"
        )
    elif model == "Gemini 2.5 Flash":
        return ChatGoogleGenerativeAI(temperature=temperature, model="gemini-2.5-flash")


# ============================================================
# [수정] 에이전트 생성 방식 변경
# - 기존: create_tool_calling_agent + AgentExecutor 조합 (LangChain 0.x)
# - 변경: create_agent 단일 API (LangChain 1.0+)
# - 이유: 코드 간소화 + checkpointer 기반 상태 관리 + 미들웨어 지원
# ============================================================
def create_customer_support_agent():
    tools = [fetch_qa_content, fetch_stores_by_prefecture]
    custom_system_prompt = load_system_prompt("./prompt/system_prompt.txt")
    llm = select_model()

    # [수정] SummarizationMiddleware 추가
    # - 대화가 길어지면 자동으로 이전 대화 내용을 요약
    summarization_middleware = SummarizationMiddleware(
        model=llm,
        max_tokens_before_summary=8000,
        messages_to_keep=10,
    )

    # [수정] create_agent 사용 (system_prompt 직접 전달, checkpointer 사용)
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=custom_system_prompt,
        checkpointer=st.session_state["checkpointer"],
        middleware=[summarization_middleware],
        debug=True
    )

    return agent


# ============================================================
# Main Function
# - [수정] StreamlitLanggraphHandler 사용 (기존 StreamlitCallbackHandler 대체)
# ============================================================
def main():
    init_page()
    init_messages()
    customer_support_agent = create_customer_support_agent()

    cache = Cache()

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="법인 명의로 계약이 가능한가요?"):
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 첫 번째 질문인 경우 캐시 확인
        if st.session_state["first_question"]:
            if cache_content := cache.search(query=prompt):
                st.chat_message("assistant").write(f"(cache) {cache_content}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": cache_content}
                )
                st.stop()

        with st.chat_message("assistant"):
            # [수정] StreamlitLanggraphHandler 사용 (기존 StreamlitCallbackHandler 대체)
            handler = StreamlitLanggraphHandler(
                container=st.container(),
                expand_new_thoughts=True,
                max_thought_containers=4,
            )

            # [수정] 에이전트 호출 방식 변경
            response = handler.invoke(
                agent=customer_support_agent,
                input={"messages": [{"role": "user", "content": prompt}]},
                config={"configurable": {"thread_id": st.session_state["thread_id"]}}
            )

            if response:
                st.session_state.messages.append({"role": "assistant", "content": response})

        # 첫 번째 질문인 경우 캐시에 저장
        if st.session_state["first_question"] and response:
            cache.save(prompt, response)


if __name__ == "__main__":
    main()
