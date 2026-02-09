import streamlit as st
import uuid  # thread_id 생성용

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

# Custom Streamlit Handler
from youngjin_langchain_tools import StreamlitLanggraphHandler


@st.cache_data
def load_system_prompt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

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

def create_customer_support_agent():
    tools = [fetch_qa_content, fetch_stores_by_prefecture]
    custom_system_prompt = load_system_prompt("./prompt/system_prompt.txt")
    llm = select_model()

    summarization_middleware = SummarizationMiddleware(
        model=llm,
        max_tokens_before_summary=8000,
        messages_to_keep=10,
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=custom_system_prompt,
        checkpointer=st.session_state["checkpointer"],
        middleware=[summarization_middleware],
        debug=True
    )

    return agent


def run_agent(agent, user_input, handler, thread_id):
    response = handler.invoke(
        agent=agent,
        input={"messages": [{"role": "user", "content": user_input}]},
        config={"configurable": {"thread_id": thread_id}}
    )
    return response



def main():
    init_page()
    init_messages()

    if "run_id" not in st.session_state:
        st.session_state["run_id"] = None

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
                with st.chat_message("assistant"):
                    st.write(f"(cache) {cache_content}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": cache_content}
                )
                st.stop()

        with st.chat_message("assistant"):
            handler = StreamlitLanggraphHandler(
                container=st.container(),
                expand_new_thoughts=True,
                max_thought_containers=4,
            )

            response = run_agent(
                customer_support_agent,
                prompt,
                handler,
                st.session_state["thread_id"]
            )

            if hasattr(handler, 'run_id'):
                st.session_state["run_id"] = str(handler.run_id)
                print("🔥 RUN ID GENERATED:", st.session_state["run_id"])

            if response:
                st.session_state.messages.append({"role": "assistant", "content": response})

        # 첫 번째 질문인 경우 캐시에 저장
        if st.session_state["first_question"] and response:
            cache.save(prompt, response)

    # LangSmith feedback 버튼
    add_feedback()


if __name__ == "__main__":
    main()
