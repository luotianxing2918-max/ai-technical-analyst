import re


SUPPORTED_SOURCES = {"local_rag", "web", "web_reader"}
SOURCE_ORDER = ("local_rag", "web", "web_reader")
URL_PATTERN = re.compile(r"https?://[^\s)\]}>]+")
ENGLISH_TERM_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b")

KNOWN_ENGLISH_TERMS = (
    "AgentBench",
    "SWE-bench",
    "GAIA",
    "LangGraph",
    "CrewAI",
    "AutoGen",
    "Semantic Kernel",
    "DenseFieldNet",
    "MCP",
)

KNOWN_CHINESE_TERMS = (
    "AI Agent",
    "智能体",
    "模型上下文协议",
    "核心架构",
    "评测方法",
    "图像恢复",
    "目标检测",
)

COMPARISON_MARKERS = (
    "比较",
    "对比",
    "区别",
    "差异",
    "特点",
    "compare",
    "comparison",
    "contrast",
    "difference",
)


SOURCE_TYPES = {
    "local_rag": ("local_document", "high"),
    "web": ("search_result", "low"),
    "web_reader": ("web_page", "medium"),
}


def _normalise_term(term):
    return re.sub(r"\s+", " ", term.strip()).casefold()


def _extract_question_terms(user_question):
    question = str(user_question or "")
    terms = []

    for term in KNOWN_CHINESE_TERMS:
        if term.casefold() in question.casefold():
            terms.append(term)

    question_lower = question.casefold()
    for term in KNOWN_ENGLISH_TERMS:
        if term.casefold() in question_lower:
            terms.append(term)

    # Recognise additional capitalised technical names without treating every
    # ordinary English word in the question as an evidence term.
    for token in ENGLISH_TERM_PATTERN.findall(question):
        if len(token) >= 4 and (
            any(char.isupper() for char in token[1:])
            or "-" in token
        ):
            terms.append(token)

    unique_terms = []
    seen = set()
    for term in terms:
        key = _normalise_term(term)
        if key not in seen:
            seen.add(key)
            unique_terms.append(term)
    return unique_terms


def _is_comparison_question(user_question):
    question_lower = str(user_question or "").casefold()
    return any(marker.casefold() in question_lower for marker in COMPARISON_MARKERS)


def check_evidence_coverage(user_question, verified_evidence) -> dict:
    """检查有效 Evidence 是否覆盖用户问题中的核心术语。"""
    terms = _extract_question_terms(user_question)
    evidence_text = "\n".join(
        str(item.get("content") or "")
        for item in verified_evidence or []
    ).casefold()

    covered_terms = [
        term for term in terms
        if _normalise_term(term) in evidence_text
    ]
    missing_terms = [term for term in terms if term not in covered_terms]
    coverage_ratio = (
        len(covered_terms) / len(terms)
        if terms
        else 1.0
    )
    warnings = []
    if missing_terms and _is_comparison_question(user_question):
        warnings.append("Evidence 未覆盖全部比较对象，当前证据不足以完成完整比较。")

    return {
        "covered_terms": covered_terms,
        "missing_terms": missing_terms,
        "coverage_ratio": coverage_ratio,
        "warnings": warnings,
    }


def _extract_urls(content):
    urls = []
    for url in URL_PATTERN.findall(str(content or "")):
        if url not in urls:
            urls.append(url)
    return urls


def _collect_successful_evidence(observations):
    evidence_by_source = {source: [] for source in SOURCE_ORDER}
    failures = []

    for observation in observations or []:
        tool_result = observation.get("tool_result") or {}
        if tool_result.get("success") is True:
            source = tool_result.get("source")
            if source in SUPPORTED_SOURCES:
                evidence_by_source[source].append(observation)
            else:
                failures.append((observation, "不支持的 evidence source。"))
        else:
            failures.append((observation, "工具执行失败。"))

    evidence = [
        observation
        for source in SOURCE_ORDER
        for observation in evidence_by_source[source]
    ]
    return evidence, failures


def verify_observations(observations, user_question="") -> dict:
    """检查 Evidence 的可追溯性和来源完整性，不验证事实语义。"""
    evidence, failures = _collect_successful_evidence(observations)
    warnings = []
    verified_evidence = []
    seen_urls = set()

    for observation, reason in failures:
        step = observation.get("step", "")
        warnings.append(f"忽略失败或不支持的 observation（step {step}）：{reason}")

    for evidence_id, observation in enumerate(evidence, start=1):
        tool_result = observation.get("tool_result") or {}
        source = tool_result.get("source")
        source_type, source_quality = SOURCE_TYPES[source]
        content = tool_result.get("content")
        content_text = str(content or "").strip()
        metadata = tool_result.get("metadata") or {}
        urls = []

        if not content_text:
            warnings.append(f"Evidence {evidence_id} 内容为空。")

        if source == "web_reader":
            url = metadata.get("url")
            if isinstance(url, str) and URL_PATTERN.match(url.strip()):
                urls = [url.strip()]
            else:
                warnings.append(f"Evidence {evidence_id} 的 Web Reader metadata 中缺少有效 URL。")
        elif source == "web":
            urls = _extract_urls(content_text)
            if not urls:
                warnings.append(f"Evidence {evidence_id} 的 Web Search 结果中缺少 URL。")

        for url in urls:
            if url in seen_urls:
                warnings.append(f"重复来源 URL: {url}")
            else:
                seen_urls.add(url)

        verified_evidence.append({
            "evidence_id": evidence_id,
            "source": source,
            "source_type": source_type,
            "source_quality": source_quality,
            "step": observation.get("step"),
            "action": observation.get("action"),
            "query": observation.get("query"),
            "content": content,
            "metadata": metadata,
            "urls": urls,
        })

    expected_ids = list(range(1, len(verified_evidence) + 1))
    actual_ids = [item["evidence_id"] for item in verified_evidence]
    if actual_ids != expected_ids:
        warnings.append("Evidence 编号不连续或不唯一。")

    coverage = check_evidence_coverage(user_question, verified_evidence)
    warnings.extend(coverage["warnings"])

    return {
        "success": not warnings,
        "verified_evidence": verified_evidence,
        "warnings": warnings,
        "coverage": coverage,
        "summary": (
            f"检查完成：{len(verified_evidence)} 条成功 Evidence，"
            f"{len(warnings)} 条 warning。"
        ),
    }


if __name__ == "__main__":
    mock_observations = [
        {
            "step": 1,
            "action": "RAG",
            "query": "model architecture",
            "tool_result": {
                "success": True,
                "source": "local_rag",
                "content": "Local project evidence.",
                "metadata": {},
            },
        },
        {
            "step": 2,
            "action": "WEB_READ",
            "query": "https://example.com/article",
            "tool_result": {
                "success": True,
                "source": "web_reader",
                "content": "Web page evidence.",
                "metadata": {"url": "https://example.com/article"},
            },
        },
        {
            "step": 3,
            "action": "WEB_READ",
            "query": "missing-url",
            "tool_result": {
                "success": True,
                "source": "web_reader",
                "content": "Page without metadata URL.",
                "metadata": {},
            },
        },
        {
            "step": 4,
            "action": "SEARCH",
            "query": "duplicate source",
            "tool_result": {
                "success": True,
                "source": "web",
                "content": "【URL】: https://example.com/article",
                "metadata": {},
            },
        },
        {
            "step": 5,
            "action": "SEARCH",
            "query": "failed search",
            "tool_result": {
                "success": False,
                "source": "web",
                "error": "未找到相关结果。",
            },
        },
    ]
    print(verify_observations(mock_observations))
