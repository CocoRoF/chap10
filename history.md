1.

# 🔧 LangSmith 트레이싱을 위한 코드 변경 요약

## ✅ 1) collect_runs import 변경

```diff
- from langchain import callbacks
+ from langchain_core.tracers.context import collect_runs
```

**이유:**
트레이싱 기능이 `langchain` → `langchain_core` 로 이동함.

---

## ✅ 2) collect_runs 호출 위치 변경

```diff
- with callbacks.collect_runs() as cb:
+ with collect_runs() as cb:
```

**이유:**
최신 버전에서는 `collect_runs()` 를 `langchain_core` 에서 직접 호출해야 LangSmith로 Run이 전송됨.

---

## ✅ 3) callbacks 모듈 제거

```diff
- from langchain import callbacks
```

**이유:**
더 이상 트레이싱 기능을 제공하지 않음.

---

# 🎉 한 줄 요약

**트레이싱 기능이 langchain → langchain_core 로 이동했기 때문에, import와 collect_runs 호출 경로만 최신 구조로 수정한 것.**


2.
setx LANGCHAIN_TRACING_V2 "true"
setx LANGCHAIN_ENDPOINT "https://api.smith.langchain.com"
setx LANGCHAIN_API_KEY "lsv2_pt_~~~~"
setx LANGCHAIN_PROJECT "youngjin_mobile"

$env:LANGCHAIN_TRACING_V2="true"
$env:LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
$env:LANGCHAIN_API_KEY="lsv2_pt_~~~~"
$env:LANGCHAIN_PROJECT="youngjin_mobile"
