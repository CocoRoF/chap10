# GitHub: https://github.com/naotaka1128/llm_app_codes/chapter_010/tools/fetch_stores_by_prefecture.py

from pathlib import Path
import pandas as pd
import streamlit as st
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# 스크립트 파일 위치 기준으로 프로젝트 루트 경로 계산
BASE_DIR = Path(__file__).resolve().parent.parent


class FetchStoresInput(BaseModel):
    """타입을 지정하기 위한 클래스"""

    pref: str = Field()


@st.cache_data(ttl="1d", show_spinner=False)
def load_stores_from_csv():
    csv_path = BASE_DIR / "data" / "bearmobile_stores.csv"
    df = pd.read_csv(csv_path)
    return df.sort_values(by="pref_id")


@tool(args_schema=FetchStoresInput)
def fetch_stores_by_prefecture(pref):
    """
        지역별로 지점을 검색하는 도구입니다.

        이 도구는 다음 데이터를 포함한 지점 목록을 반환합니다:
    - `store_name`（지점명）
    - `postal_code`（우편번호）
    - `address`（주소）
    - `tel`（전화번호）

        전국의 지점 목록이 필요한 경우, '전국'이라고 입력해서 검색하세요.
        다만, 이 검색 방법은 권장하지 않습니다.
        사용자가 '어디에 지점이 있나요?'라고 물어온 경우에는,
          먼저 사용자의 거주 지역명을 확인해 주세요.

        빈 리스트가 반환된 경우는 해당 지역에 지점이 없다는 의미입니다.
        이럴 때는 사용자에게 질문 내용을 더 명확하게 해달라고 요청하는 것이 좋습니다.

    Returns
    -------
    List[Dict[str, Any]]:
    - store_name: str
    - post_code: str
    - address: str
    - tel: str
    """
    df = load_stores_from_csv()
    if pref != "전국":
        df = df[df["pref"] == pref]
    return [
        {
            "store_name": row["name"],
            "post_code": row["post_code"],
            "address": row["address"],
            "tel": row["tel"],
        }
        for _, row in df.iterrows()
    ]
