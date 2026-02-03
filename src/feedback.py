import streamlit as st
from langsmith import Client
from streamlit_feedback import streamlit_feedback


def add_feedback():
    langsmith_client = Client()

    run_id = st.session_state.get("run_id")
    if not run_id:
        st.info("대화를 시작하면 응답에 대한 피드백을 남길 수 있습니다.")
        return

    # 피드백 가져오기
    feedback = streamlit_feedback(
        feedback_type="thumbs",
        optional_text_label="[선택] 설명을 입력해 주세요",
        key=f"feedback_{run_id}",
    )
    print("👍 RAW FEEDBACK DATA:", feedback)

    scores = {"👍": 1, "👎": 0}

    if feedback:
        # 선택된 피드백 옵션에 따른 점수 가져오기
        score = scores.get(feedback["score"])

        if score is not None:
            # 선택된 옵션과 점수를 사용해 피드백 타입 문자열 생성
            feedback_type_str = f"thumbs {feedback['score']}"

            # 생성된 피드백 타입 문자열과 선택 입력된 코멘트를 사용하여
            # 피드백을 기록
            feedback_record = langsmith_client.create_feedback(
                run_id,
                feedback_type_str,
                score=score,
                comment=feedback.get("text"),
            )
            print("📩 FEEDBACK SENT TO LANGSMITH:", feedback_record)
            # 피드백 ID와 점수를 세션 상태에 저장
            st.session_state.feedback = {
                "feedback_id": str(feedback_record.id),
                "score": score,
            }
        else:
            # 유효하지 않은 피드백 점수인 경우 경고 표시
            st.warning("유효하지 않은 피드백 점수입니다.")
