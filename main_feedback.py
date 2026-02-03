# GitHub: https://github.com/naotaka1128/llm_app_codes/chapter_010/main_feedback_new.py

import streamlit as st
import uuid  #  thread_id 생성용

# ============================================================
#  LangChain 1.0.0+ 신규 create_agent 사용
# ============================================================
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

# models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# custom tools
from tools.fetch_qa_content import fetch_qa_content
from tools.fetch_stores_by_prefecture import fetch_stores_by_prefecture

# cache / feedback
from src.cache import Cache
from src.feedback import add_feedback

#  StreamlitLanggraphHandler 사용 (기존 StreamlitCallbackHandler 대체)
from youngjin_langchain_tools import StreamlitLanggraphHandler

# LangSmith trace
from langsmith import traceable

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
        #  ConversationBufferWindowMemory 대신 InMemorySaver + thread_id 사용
        st.session_state["checkpointer"] = InMemorySaver()
        st.session_state["thread_id"] = str(uuid.uuid4())

    st.session_state["first_question"] = len(st.session_state.messages) == 1


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
#  에이전트 생성 방식 변경
# (create_tool_calling_agent + AgentExecutor → create_agent)
# ============================================================
def create_customer_support_agent():
    tools = [fetch_qa_content, fetch_stores_by_prefecture]
    custom_system_prompt = load_system_prompt("./prompt/system_prompt.txt")
    llm = select_model()

    #  SummarizationMiddleware 추가
    summarization_middleware = SummarizationMiddleware(
        model=llm,
        max_tokens_before_summary=8000,
        messages_to_keep=10,
    )

    #  create_agent 사용 (system_prompt 직접 전달, checkpointer 사용)
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
#  run_agent 함수 변경 - StreamlitLanggraphHandler 사용
# ============================================================
@traceable  # LangSmith 트레이스
def run_agent(agent, user_input, handler, thread_id):
    """에이전트 실행 및 응답 반환 (LangSmith traceable)"""
    response = handler.invoke(
        agent=agent,
        input={"messages": [{"role": "user", "content": user_input}]},
        config={"configurable": {"thread_id": thread_id}}
    )
    return response


# ============================================================
# Main Function - StreamlitLanggraphHandler 사용
# ============================================================
def main():
    init_page()
    init_messages()

    if "run_id" not in st.session_state:
        st.session_state["run_id"] = None

    customer_support_agent = create_customer_support_agent()
    cache = Cache()

    #  대화 히스토리 표시 방식 변경
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="법인 명의로 계약이 가능한가요?"):
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 첫 번째 질문인 경우 캐시 확인
        if st.session_state["first_question"]:
            if cache_content := cache.search(query=prompt):
                with st.chat_message("assistant"):
                    st.write(f"(cache) {cache_content}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": cache_content}
                )
                st.stop()

        with st.chat_message("assistant"):
            #  StreamlitLanggraphHandler 사용 (기존 StreamlitCallbackHandler 대체)
            handler = StreamlitLanggraphHandler(
                container=st.container(),
                expand_new_thoughts=True,
                max_thought_containers=4,
            )

            #  에이전트 호출 방식 변경
            response = run_agent(
                customer_support_agent,
                prompt,
                handler,
                st.session_state["thread_id"]
            )

            # [NOTE] run_id는 LangSmith traceable 데코레이터에서 자동 생성됨
            # 필요시 handler 또는 langsmith 클라이언트에서 run_id 가져오기
            if hasattr(handler, 'run_id'):
                st.session_state["run_id"] = str(handler.run_id)
                print("🔥 RUN ID GENERATED:", st.session_state["run_id"])

            # 응답 저장
            if response:
                st.session_state.messages.append({"role": "assistant", "content": response})

        # 첫 번째 질문이면 캐시에 저장
        if st.session_state["first_question"] and response:
            cache.save(prompt, response)

    # LangSmith feedback 버튼 유지
    add_feedback()


if __name__ == "__main__":
    main()
