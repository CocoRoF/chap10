# GitHub: https://github.com/naotaka1128/llm_app_codes/chapter_010/main_cache_new.py

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
from src.cache import Cache

#  StreamlitLanggraphHandler 사용 (기존 StreamlitCallbackHandler 대체)
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


@st.cache_data  # 캐시를 사용하도록 변경
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

    if len(st.session_state.messages) == 1:  # 환영 메시지뿐인 경우
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
#  에이전트 생성 방식 변경
# (create_tool_calling_agent + AgentExecutor → create_agent)
# ============================================================
def create_customer_support_agent():
    tools = [fetch_qa_content, fetch_stores_by_prefecture]
    # 캐시를 사용하도록 변경
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
# Main Function - StreamlitLanggraphHandler 사용
# ============================================================
def main():
    init_page()
    init_messages()
    customer_support_agent = create_customer_support_agent()

    # 캐시 초기화
    cache = Cache()

    #  대화 히스토리 표시 방식 변경
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="법인 명의로 계약이 가능한가요?"):
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 처음 질문인 경우 캐시를 확인
        if st.session_state["first_question"]:
            if cache_content := cache.search(query=prompt):
                st.chat_message("assistant").write(f"(cache) {cache_content}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": cache_content}
                )
                st.stop()  # 캐시 내용을 출력한 경우 실행 종료

        with st.chat_message("assistant"):
            #  StreamlitLanggraphHandler 사용 (기존 StreamlitCallbackHandler 대체)
            handler = StreamlitLanggraphHandler(
                container=st.container(),
                expand_new_thoughts=True,
                max_thought_containers=4,
            )

            #  에이전트 호출 방식 변경
            response = handler.invoke(
                agent=customer_support_agent,
                input={"messages": [{"role": "user", "content": prompt}]},
                config={"configurable": {"thread_id": st.session_state["thread_id"]}}
            )

            # 응답 저장
            if response:
                st.session_state.messages.append({"role": "assistant", "content": response})

        # 처음 질문인 경우 캐시에 저장
        if st.session_state["first_question"] and response:
            cache.save(prompt, response)


if __name__ == "__main__":
    main()
