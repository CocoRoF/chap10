# 스트리밍 테스트용 기본 에이전트

import streamlit as st
import uuid

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

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

SYSTEM_PROMPT = "You are a helpful assistant."


def init_page():
    st.set_page_config(page_title="Basic Agent", page_icon="🤖")
    st.header("Basic Agent 🤖")
    st.sidebar.title("옵션")


def init_messages():
    clear_button = st.sidebar.button("대화 초기화", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 무엇이든 물어보세요."}]
        st.session_state["checkpointer"] = InMemorySaver()
        st.session_state["thread_id"] = str(uuid.uuid4())


def select_model(temperature=0):
    models = ("GPT-4o mini", "GPT-4o", "Claude Sonnet 4", "Gemini 2.0 Flash")
    model = st.sidebar.radio("사용할 모델 선택:", models)
    if model == "GPT-4o mini":
        return ChatOpenAI(temperature=temperature, model="gpt-4o-mini")
    elif model == "GPT-4o":
        return ChatOpenAI(temperature=temperature, model="gpt-4o")
    elif model == "Claude Sonnet 4":
        return ChatAnthropic(temperature=temperature, model="claude-sonnet-4-20250514")
    elif model == "Gemini 2.0 Flash":
        return ChatGoogleGenerativeAI(temperature=temperature, model="gemini-2.0-flash")


def create_basic_agent():
    llm = select_model()

    agent = create_agent(
        model=llm,
        tools=[],  # 툴 없음
        system_prompt=SYSTEM_PROMPT,
        checkpointer=st.session_state["checkpointer"],
    )

    return agent


def main():
    init_page()
    init_messages()
    agent = create_basic_agent()

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="아무거나 물어보세요"):
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            handler = StreamlitLanggraphHandler(
                container=st.container(),
                expand_new_thoughts=True,
                max_thought_containers=4,
            )

            response = handler.invoke(
                agent=agent,
                input={"messages": [{"role": "user", "content": prompt}]},
                config={"configurable": {"thread_id": st.session_state["thread_id"]}}
            )

            if response:
                st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
