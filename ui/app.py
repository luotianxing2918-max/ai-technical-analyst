import sys
from pathlib import Path

import streamlit as st


APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "app"))

from agent import run_agent_once  # noqa: E402


st.set_page_config(
    page_title="AI Technical Analyst",
    page_icon="🔎",
    layout="wide",
)

st.title("AI Technical Analyst")
st.caption("Technical Research & Evidence Analysis Agent")

examples = [
    "根据我的本地 X-ray 项目资料，分析 DenseFieldNet 的核心方法，然后搜索相关研究进行比较；请给出来源，并在证据不足时明确说明。",
    "比较 LangGraph、CrewAI、AutoGen 和 Semantic Kernel。",
    "分析 MCP 的核心架构及其与传统 API 调用的区别。",
]

with st.sidebar:
    st.subheader("示例问题")
    selected_example = st.selectbox("选择一个示例", ["不使用示例"] + examples)

default_question = selected_example if selected_example != "不使用示例" else ""
question = st.text_area(
    "技术问题",
    value=default_question,
    height=150,
    placeholder="请输入技术问题……",
)


def render_event(event):
    status = event.get("status", "")
    icon = "⏳" if status == "running" else ("✅" if event.get("success") else "⚠️")
    query = event.get("query") or "生成最终报告"
    return f"{icon} Step {event.get('step')}: {event.get('action')} · {query}"


if st.button("开始分析", type="primary", disabled=not question.strip()):
    event_placeholder = st.empty()
    events = []

    def on_event(event):
        events.append(event)
        with event_placeholder.container():
            st.subheader("Agent Workflow")
            for item in events:
                st.write(render_event(item))
            if events and events[-1].get("status") == "running":
                st.progress(min(len(events) / 10, 0.95))

    try:
        result = run_agent_once(question.strip(), event_callback=on_event)
    except Exception as error:
        st.error(
            "分析未完成。请确认 Ollama 已启动、Router/Final 模型已安装，"
            f"并检查本地知识库配置。错误：{error}"
        )
        st.stop()

    st.success("分析完成")

    st.subheader("Final Report")
    st.markdown(result["final_answer"] or "未生成最终报告。")

    verification = result["verification"]
    coverage = verification.get("coverage", {})
    with st.expander("Evidence Verification", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Verified Evidence", len(verification.get("verified_evidence", [])))
        col2.metric("Evidence Coverage", f"{coverage.get('coverage_ratio', 0.0):.0%}")
        col3.metric("Warnings", len(verification.get("warnings", [])))
        st.write("Covered terms:", ", ".join(coverage.get("covered_terms", [])) or "无")
        st.write("Missing terms:", ", ".join(coverage.get("missing_terms", [])) or "无")
        for warning in verification.get("warnings", []):
            st.warning(warning)

    st.subheader("Evidence / Sources")
    verified_by_id = {
        item["evidence_id"]: item
        for item in verification.get("verified_evidence", [])
    }
    for evidence_id, evidence in verified_by_id.items():
        observation = next(
            (
                item
                for item in result["observations"]
                if item.get("step") == evidence.get("step")
            ),
            {},
        )
        tool_result = observation.get("tool_result") or {}
        label = f"Evidence {evidence_id} · {tool_result.get('source', 'unknown')}"
        with st.expander(label):
            metadata = tool_result.get("metadata") or {}
            url = metadata.get("url")
            st.write("Source type:", evidence.get("source_type", "unknown"))
            if url:
                st.write("URL:", url)
            st.write("Status: success")
            st.write(str(tool_result.get("content", ""))[:1200])

    failures = [
        item for item in result["observations"]
        if not (item.get("tool_result") or {}).get("success")
    ]
    for observation in failures:
        error = (observation.get("tool_result") or {}).get("error", "未知错误")
        st.warning(
            f"Step {observation.get('step')} {observation.get('action')} 失败：{error}"
        )