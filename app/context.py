import json
import re


SUPPORTED_SOURCES = {"local_rag", "web", "web_reader"}
SOURCE_ORDER = ("local_rag", "web", "web_reader")
MAX_CONTENT_LENGTH = 8000
URL_PATTERN = re.compile(r"https?://[^\s)\]}>]+")


def _format_metadata(metadata):
    if not metadata:
        return "{}"
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True)


def _collect_evidence(observations):
    evidence_by_source = {source: [] for source in SOURCE_ORDER}
    failures = []

    for observation in observations or []:
        tool_result = observation.get("tool_result") or {}
        if tool_result.get("success") is True:
            source = tool_result.get("source")
            if source in SUPPORTED_SOURCES:
                evidence_by_source[source].append((observation, tool_result))
        else:
            failures.append(observation)

    evidence = [
        item
        for source in SOURCE_ORDER
        for item in evidence_by_source[source]
    ]
    return evidence, failures


def _extract_search_urls(content):
    urls = []
    for url in URL_PATTERN.findall(str(content or "")):
        if url not in urls:
            urls.append(url)
    return urls


def _format_evidence(index, observation, tool_result):
    source = tool_result.get("source", "unknown")
    content = str(tool_result.get("content", ""))
    content = content[:MAX_CONTENT_LENGTH]

    lines = [
        f"[Evidence {index}]",
        f"Source: {source}",
        f"Step: {observation.get('step', '')}",
        f"Action: {observation.get('action', '')}",
        f"Query: {observation.get('query', '')}",
    ]
    if source == "web":
        lines.append("Type: candidate search summaries")
        search_urls = _extract_search_urls(content)
        if search_urls:
            lines.append(f"URLs: {', '.join(search_urls)}")
    elif source == "web_reader":
        lines.append("Type: webpage正文")

    metadata = tool_result.get("metadata", {})
    if source == "web_reader" and metadata.get("url"):
        lines.append(f"URL: {metadata['url']}")
    lines.extend([
        f"Metadata: {_format_metadata(metadata)}",
        "Content:",
        content,
    ])
    return "\n".join(lines)


def format_source_index(observations):
    """根据实际成功 Evidence 生成最终报告的参考来源列表。"""
    evidence, _ = _collect_evidence(observations)
    lines = ["REFERENCE SOURCES:"]

    for index, (observation, tool_result) in enumerate(evidence, start=1):
        source = tool_result.get("source")
        if source == "local_rag":
            description = "Local RAG：本地知识库"
        elif source == "web_reader":
            url = (tool_result.get("metadata") or {}).get("url")
            description = f"Web Reader：{url}" if url else "Web Reader：网页正文"
        else:
            urls = _extract_search_urls(tool_result.get("content", ""))
            description = (
                f"Web Search：{', '.join(urls)}"
                if urls
                else "Web Search：候选来源摘要"
            )
        lines.append(f"- [Evidence {index}] {description}")

    if not evidence:
        lines.append("- 无可用 Evidence")

    return "\n".join(lines)


def _format_failures(observations):
    lines = ["ERRORS:"]
    for observation in observations:
        tool_result = observation.get("tool_result") or {}
        lines.extend([
            f"[Error Step {observation.get('step', '')}]",
            f"Source: {tool_result.get('source', observation.get('action', '').lower())}",
            f"Action: {observation.get('action', '')}",
            f"Query: {observation.get('query', '')}",
            f"Error: {tool_result.get('error', '工具未返回有效内容。')}",
        ])
    return "\n".join(lines)


def build_context(user_question, observations):
    """将 Agent observations 整理为供 Final LLM 使用的纯文本 Context。"""
    evidence, failures = _collect_evidence(observations)

    lines = ["USER QUESTION:", str(user_question), "", "EVIDENCE:"]
    if evidence:
        for index, (observation, tool_result) in enumerate(evidence, start=1):
            lines.extend([_format_evidence(index, observation, tool_result), ""])
    else:
        lines.append("No successful tool evidence was available.")
        if failures:
            lines.extend(["", _format_failures(failures)])

    return "\n".join(lines).rstrip()