# GitHub: https://github.com/naotaka1128/llm_app_codes/chapter_010/main_new.py

import streamlit as st
import uuid  #  thread_id 생성용

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

# ============================================================
# [수정] 스트리밍 핸들러 변경
# - 기존: StreamlitCallbackHandler (LangChain 레거시)
# - 변경: StreamlitLanggraphHandler (LangGraph 호환)
# - 이유: create_agent가 내부적으로 LangGraph 기반으로 동작하므로
#         LangGraph 전용 핸들러 사용 필요
# ============================================================
from youngjin_langchain_tools import StreamlitLanggraphHandler

###### dotenv 을 사용하지 않는 경우는 삭제해주세요 ######
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


# ============================================================
# System Prompt
# ============================================================
CUSTOM_SYSTEM_PROMPT = """
당신은 일본의 저가 통신사 '영진모바일'의 고객센터(CS) 상담원입니다.
고객의 문의에 대해 성실하고 정확하게 답변해주세요.

통신사 CS로서, 회사의 서비스와 휴대전화에 관한 일반적인 정보에만 답변해야 합니다.
그 외의 주제에 관한 질문에는 정중하게 답변을 거절해주세요.

답변의 정확성을 보장하기 위해, '영진모바일'에 대한 질문을 받을 경우
반드시 툴을 사용해 답을 찾아주세요.

고객이 질문에 사용한 언어로 답변해주세요.
예를 들어 영어로 질문하면 영어로, 스페인어로 질문하면 스페인어로 답변해야 합니다.

답변 과정에서 불분명한 부분이 있다면 반드시 고객에게 확인해 주세요.
그렇게 해야 고객의 진짜 의도를 정확하게 파악하고 올바른 답변을 제공할 수 있습니다.

예를 들어 고객이 "매장은 어디에 있나요?"라고 질문한 경우,
먼저 고객이 거주하는 도도부현(지역)을 물어보세요.

일본 전국의 매장 위치를 알고 싶은 고객은 거의 없습니다.
고객은 자기 지역의 매장을 알고 싶은 것입니다.
따라서 전국 매장을 검색해 답변하는 일이 없도록 하며,
고객의 의도를 완전히 파악하기 전까지는 섣불리 답변하지 마세요!

위는 한 가지 예시에 불과합니다.
다른 경우에도 항상 고객의 의도를 파악하고 적절한 답변을 해주세요.
"""


# ============================================================
# Streamlit UI Functions
# ============================================================
def init_page():
    st.set_page_config(page_title="고객센터", page_icon="🐻")
    st.header("고객센터🐻")
    st.sidebar.title("옵션")


def init_messages():
    clear_button = st.sidebar.button("대화 초기화", key="clear")
    if clear_button or "messages" not in st.session_state:
        welcome_message = (
            "영진모바일 고객센터에 오신 것을 환영합니다. 무엇이든 문의해주세요🐻"
        )
        st.session_state.messages = [{"role": "assistant", "content": welcome_message}]
        # [수정] 메모리 관리 방식 변경
        # - 기존: ConversationBufferWindowMemory (LangChain 레거시)
        # - 변경: InMemorySaver + thread_id 조합 (LangGraph 방식)
        # - 이유: create_agent는 LangGraph 기반으로 동작하며, checkpointer를 통해 대화 상태를 관리함
        st.session_state["checkpointer"] = InMemorySaver()
        st.session_state["thread_id"] = str(uuid.uuid4())


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
        system_prompt=CUSTOM_SYSTEM_PROMPT,
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

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="법인 명의로도 계약할 수 있어?"):
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            # [수정] StreamlitLanggraphHandler 사용 (기존 StreamlitCallbackHandler 대체)
            handler = StreamlitLanggraphHandler(
                container=st.container(),
                expand_new_thoughts=True,
                max_thought_containers=4,
            )

            # [수정] 에이전트 호출 방식 변경
            # - 기존: executor.invoke({"input": prompt})
            # - 변경: handler.invoke(agent, input, config)
            response = handler.invoke(
                agent=customer_support_agent,
                input={"messages": [{"role": "user", "content": prompt}]},
                config={"configurable": {"thread_id": st.session_state["thread_id"]}}
            )

            if response:
                st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
