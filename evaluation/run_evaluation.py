"""Run a repeatable, engineering-style Baseline vs Agent evaluation."""

import argparse
import json
import multiprocessing
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "app"
sys.path.insert(0, str(APP_ROOT))

from agent import FINAL_MODEL, ROUTER_MODEL, run_agent_once  # noqa: E402
from llm import call_llm  # noqa: E402


BASELINE_MODEL = FINAL_MODEL
CITATION_PATTERN = re.compile(r"\[Evidence\s+(\d+)\]")
STATUS_SUCCESS = "success"
STATUS_TIMEOUT = "timeout"
STATUS_TOOL_FAILURE = "tool_failure"
STATUS_EXCEPTION = "exception"
FAILURE_STATUSES = {STATUS_TIMEOUT, STATUS_TOOL_FAILURE, STATUS_EXCEPTION}

BASELINE_SYSTEM_PROMPT = """You are a technical analyst answering directly without external tools.
Answer the user's question clearly and honestly. Do not invent sources, URLs,
or Evidence citations. If the question asks for information that is not
provided in the prompt, state uncertainty instead of pretending to have verified it.
Use Markdown when useful."""


def _load_cases(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _empty_result(status, latency=0.0, error=None):
    return {
        "answer": "",
        "observations": [],
        "verification": {},
        "events": [],
        "status": status,
        "error": error,
        "latency": latency,
    }


def _safe_call_baseline(question):
    try:
        return {
            "answer": call_llm(
                question,
                model=BASELINE_MODEL,
                system_prompt=BASELINE_SYSTEM_PROMPT,
            ),
            "error": None,
            "status": STATUS_SUCCESS,
        }
    except Exception as error:
        return {"answer": "", "error": str(error), "status": STATUS_EXCEPTION}


def _run_baseline(question):
    started = time.perf_counter()
    result = _safe_call_baseline(question)
    result["latency"] = time.perf_counter() - started
    return result


def _agent_tool_failures(observations):
    failures = []
    for observation in observations or []:
        tool_result = observation.get("tool_result") or {}
        if tool_result.get("success") is not True:
            source = tool_result.get("source", observation.get("action", "unknown"))
            error = tool_result.get("error", "tool returned an unsuccessful result")
            failures.append(f"{source}: {error}")
    return failures


def _run_agent(question):
    events = []
    started = time.perf_counter()
    try:
        result = run_agent_once(question, event_callback=events.append)
        observations = result.get("observations", [])
        failures = _agent_tool_failures(observations)
        return {
            "answer": result.get("final_answer", ""),
            "observations": observations,
            "verification": result.get("verification", {}),
            "events": events,
            "status": STATUS_TOOL_FAILURE if failures else STATUS_SUCCESS,
            "error": "; ".join(failures) if failures else None,
            "latency": time.perf_counter() - started,
        }
    except Exception as error:
        return {
            **_empty_result(STATUS_EXCEPTION),
            "events": events,
            "error": str(error),
            "latency": time.perf_counter() - started,
        }


def _evaluate_call(kind, question, result_queue):
    # rag.py uses a relative Chroma path; every worker must use the project root.
    os.chdir(PROJECT_ROOT)
    result = _run_baseline(question) if kind == "baseline" else _run_agent(question)
    result_queue.put(result)


def _run_single_call(kind, question, timeout):
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_evaluate_call,
        args=(kind, question, result_queue),
    )
    started = time.perf_counter()
    try:
        process.start()
        process.join(timeout)
    except Exception as error:
        if process.is_alive():
            process.terminate()
            process.join()
        return _empty_result(STATUS_EXCEPTION, time.perf_counter() - started, str(error))

    elapsed = time.perf_counter() - started
    if process.is_alive():
        process.terminate()
        process.join()
        return _empty_result(
            STATUS_TIMEOUT,
            elapsed,
            f"evaluation timeout after {timeout} seconds",
        )

    try:
        result = result_queue.get(timeout=1)
    except Exception:
        return _empty_result(
            STATUS_EXCEPTION,
            elapsed,
            f"worker exited with code {process.exitcode} without a result",
        )
    finally:
        result_queue.close()
        result_queue.join_thread()

    result["latency"] = elapsed
    return result


def _run_case(case, timeout):
    return {
        "baseline": _run_single_call("baseline", case["question"], timeout),
        "agent": _run_single_call("agent", case["question"], timeout),
    }


def _task_completion(answer, status):
    if status in FAILURE_STATUSES and not str(answer or "").strip():
        return None
    return 1.0 if str(answer or "").strip() else 0.0


def _citation_traceability(answer, verification):
    citations = [int(value) for value in CITATION_PATTERN.findall(str(answer or ""))]
    if not citations:
        return 0.0
    valid_ids = {
        item.get("evidence_id")
        for item in (verification or {}).get("verified_evidence", [])
    }
    return sum(citation in valid_ids for citation in citations) / len(citations)


def _evidence_sufficiency(answer, verification):
    if not verification or "coverage" not in verification:
        return None
    missing_terms = (verification.get("coverage") or {}).get("missing_terms", [])
    acknowledged = "证据不足" in str(answer or "") or "信息不足" in str(answer or "")
    return 1.0 if bool(missing_terms) == acknowledged else 0.0


def _baseline_metrics(result):
    return {
        "task_completion": _task_completion(result.get("answer"), result.get("status")),
        "evidence_coverage": None,
        "citation_traceability": None,
        "evidence_sufficiency": None,
        "latency": result.get("latency", 0.0),
    }


def _agent_metrics(result):
    verification = result.get("verification") or {}
    coverage = verification.get("coverage") or {}
    return {
        "task_completion": _task_completion(result.get("answer"), result.get("status")),
        "evidence_coverage": coverage.get("coverage_ratio"),
        "citation_traceability": _citation_traceability(result.get("answer"), verification),
        "evidence_sufficiency": _evidence_sufficiency(result.get("answer"), verification),
        "latency": result.get("latency", 0.0),
    }


def _average(values):
    values = [value for value in values if isinstance(value, (int, float))]
    return sum(values) / len(values) if values else None


def _rate(results, key, expected):
    return sum(item[key] == expected for item in results) / len(results) if results else 0.0


def _metric_average(results, side, metric):
    return _average(item["metrics"][side].get(metric) for item in results)


def _build_overall(results):
    return {
        "total_cases": len(results),
        "baseline_success_rate": _rate(results, "baseline_status", STATUS_SUCCESS),
        "agent_success_rate": _rate(results, "agent_status", STATUS_SUCCESS),
        "baseline_timeout_rate": _rate(results, "baseline_status", STATUS_TIMEOUT),
        "agent_timeout_rate": _rate(results, "agent_status", STATUS_TIMEOUT),
        "average_baseline_latency": _average(
            item["baseline_latency"] for item in results if item["baseline_status"] != STATUS_TIMEOUT
        ),
        "average_agent_latency": _average(
            item["agent_latency"] for item in results if item["agent_status"] != STATUS_TIMEOUT
        ),
        "average_agent_evidence_coverage": _metric_average(results, "agent", "evidence_coverage"),
        "citation_traceability_rate": _metric_average(results, "agent", "citation_traceability"),
        "evidence_sufficiency_rate": _metric_average(results, "agent", "evidence_sufficiency"),
    }


def _format_optional(value):
    return "N/A" if value is None else f"{value:.3f}"


def _build_summary(payload):
    results = payload["results"]
    lines = [
        "# Baseline vs Agent Evaluation",
        "",
        "Engineering metrics only; this is not an academic benchmark.",
        "Unavailable metrics are shown as N/A; status rates include every case.",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in payload["overall"].items():
        lines.append(f"| {key} | {value if key == 'total_cases' else _format_optional(value)} |")

    lines.extend([
        "",
        "## Cases",
        "",
        "| Case | Baseline status | Agent status | Baseline latency | Agent latency | Coverage | Citation traceability | Evidence sufficiency |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for result in results:
        metrics = result["metrics"]["agent"]
        lines.append(
            "| {case_id} | {baseline_status} | {agent_status} | {baseline_latency:.3f} | {agent_latency:.3f} | {coverage} | {citation} | {sufficiency} |".format(
                case_id=result["case_id"],
                baseline_status=result["baseline_status"],
                agent_status=result["agent_status"],
                baseline_latency=result["baseline_latency"],
                agent_latency=result["agent_latency"],
                coverage=_format_optional(metrics["evidence_coverage"]),
                citation=_format_optional(metrics["citation_traceability"]),
                sufficiency=_format_optional(metrics["evidence_sufficiency"]),
            )
        )

    lines.extend(["", "## Failures", ""])
    failures = [
        result for result in results
        if result["baseline_status"] != STATUS_SUCCESS or result["agent_status"] != STATUS_SUCCESS
    ]
    if not failures:
        lines.append("None")
    else:
        for result in failures:
            if result["baseline_status"] != STATUS_SUCCESS:
                lines.append(f"- {result['case_id']} baseline [{result['baseline_status']}]: {result['baseline_error']}")
            if result["agent_status"] != STATUS_SUCCESS:
                lines.append(f"- {result['case_id']} agent [{result['agent_status']}]: {result['agent_error']}")
    return "\n".join(lines) + "\n"


def run_evaluation(cases, output_dir, case_timeout):
    results = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']}: {case['question']}", flush=True)
        try:
            run_result = _run_case(case, case_timeout)
        except Exception as error:
            run_result = {
                "baseline": _empty_result(STATUS_EXCEPTION, error=str(error)),
                "agent": _empty_result(STATUS_EXCEPTION, error=str(error)),
            }

        baseline = run_result["baseline"]
        agent = run_result["agent"]
        result = {
            "case_id": case["id"],
            "category": case.get("category"),
            "question": case["question"],
            "baseline_answer": baseline.get("answer", ""),
            "agent_answer": agent.get("answer", ""),
            "baseline_status": baseline.get("status", STATUS_EXCEPTION),
            "agent_status": agent.get("status", STATUS_EXCEPTION),
            "baseline_latency": baseline.get("latency", 0.0),
            "agent_latency": agent.get("latency", 0.0),
            "observations": agent.get("observations", []),
            "verification": agent.get("verification", {}),
            "metrics": {"baseline": _baseline_metrics(baseline), "agent": _agent_metrics(agent)},
            "baseline_error": baseline.get("error"),
            "agent_error": agent.get("error"),
            "agent_events": agent.get("events", []),
        }
        results.append(result)
        print(f"  baseline={result['baseline_status']} agent={result['agent_status']}", flush=True)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline_model": BASELINE_MODEL,
        "final_model": FINAL_MODEL,
        "router_model": ROUTER_MODEL,
        "timeout_seconds": case_timeout,
        "working_directory": str(PROJECT_ROOT),
        "metric_note": "Engineering proxies only; not an academic benchmark.",
        "overall": _build_overall(results),
        "results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(_build_summary(payload), encoding="utf-8")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("test_cases.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N cases, useful for smoke tests.")
    parser.add_argument("--case-timeout", type=int, default=300, help="Maximum seconds per call before recording timeout.")
    args = parser.parse_args()
    cases = _load_cases(args.cases)
    if args.limit is not None:
        cases = cases[:args.limit]
    run_evaluation(cases, args.output_dir, args.case_timeout)


if False:
    main()

from evaluation_runner import main as stable_main

if __name__ == "__main__":
    stable_main()

'''
import argparse
import json
import multiprocessing
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "app"
sys.path.insert(0, str(APP_ROOT))

from agent import FINAL_MODEL, ROUTER_MODEL, run_agent_once  # noqa: E402
from llm import call_llm  # noqa: E402


BASELINE_MODEL = FINAL_MODEL
CITATION_PATTERN = re.compile(r"\[Evidence\s+(\d+)\]")
STATUS_SUCCESS = "success"
STATUS_TIMEOUT = "timeout"
STATUS_TOOL_FAILURE = "tool_failure"
STATUS_EXCEPTION = "exception"
FAILURE_STATUSES = {STATUS_TIMEOUT, STATUS_TOOL_FAILURE, STATUS_EXCEPTION}

BASELINE_SYSTEM_PROMPT = """You are a technical analyst answering directly without external tools.
Answer the user's question clearly and honestly. Do not invent sources, URLs,
or Evidence citations. If the question asks for information that is not
provided in the prompt, state uncertainty instead of pretending to have verified it.
Use Markdown when useful."""


def _load_cases(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _empty_result(status, latency=0.0, error=None):
    return {
        "answer": "",
        "observations": [],
        "verification": {},
        "events": [],
        "status": status,
        "error": error,
        "latency": latency,
    }


def _safe_call_baseline(question):
    try:
        return {
            "answer": call_llm(
                question,
                model=BASELINE_MODEL,
                system_prompt=BASELINE_SYSTEM_PROMPT,
            ),
            "error": None,
            "status": STATUS_SUCCESS,
        }
    except Exception as error:
        return {
            "answer": "",
            "error": str(error),
            "status": STATUS_EXCEPTION,
        }


def _run_baseline(question):
    started = time.perf_counter()
    result = _safe_call_baseline(question)
    result["latency"] = time.perf_counter() - started
    return result


def _agent_tool_failure(observations):
    failures = []
    for observation in observations or []:
        tool_result = observation.get("tool_result") or {}
        if tool_result.get("success") is not True:
            source = tool_result.get("source", observation.get("action", "unknown"))
            error = tool_result.get("error", "tool returned an unsuccessful result")
            failures.append(f"{source}: {error}")
    return failures


def _run_agent(question):
    events = []
    started = time.perf_counter()
    try:
        result = run_agent_once(question, event_callback=events.append)
        observations = result.get("observations", [])
        tool_failures = _agent_tool_failure(observations)
        return {
            "answer": result.get("final_answer", ""),
            "observations": observations,
            "verification": result.get("verification", {}),
            "events": events,
            "status": STATUS_TOOL_FAILURE if tool_failures else STATUS_SUCCESS,
            "error": "; ".join(tool_failures) if tool_failures else None,
            "latency": time.perf_counter() - started,
        }
    except Exception as error:
        return {
            **_empty_result(STATUS_EXCEPTION),
            "events": events,
            "error": str(error),
            "latency": time.perf_counter() - started,
        }


def _evaluate_call(kind, question, result_queue):
    # rag.py uses a relative Chroma path; every worker must run from the project root.
    os.chdir(PROJECT_ROOT)
    result = _run_baseline(question) if kind == "baseline" else _run_agent(question)
    result_queue.put(result)


def _run_single_call(kind, question, timeout):
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_evaluate_call,
        args=(kind, question, result_queue),
    )
    started = time.perf_counter()
    try:
        process.start()
        process.join(timeout)
    except Exception as error:
        if process.is_alive():
            process.terminate()
            process.join()
        return _empty_result(
            STATUS_EXCEPTION,
            time.perf_counter() - started,
            str(error),
        )

    elapsed = time.perf_counter() - started
    if process.is_alive():
        process.terminate()
        process.join()
        return _empty_result(
            STATUS_TIMEOUT,
            elapsed,
            f"evaluation timeout after {timeout} seconds",
        )

    try:
        result = result_queue.get(timeout=1)
    except Exception:
        return _empty_result(
            STATUS_EXCEPTION,
            elapsed,
            f"worker exited with code {process.exitcode} without a result",
        )
    finally:
        result_queue.close()
        result_queue.join_thread()

    result["latency"] = elapsed
    return result


def _run_case(case, timeout):
    return {
        "baseline": _run_single_call("baseline", case["question"], timeout),
        "agent": _run_single_call("agent", case["question"], timeout),
    }


def _task_completion(answer, status):
    if status in FAILURE_STATUSES and not str(answer or "").strip():
        return None
    return 1.0 if str(answer or "").strip() else 0.0


def _citation_traceability(answer, verification):
    citations = [int(value) for value in CITATION_PATTERN.findall(str(answer or ""))]
    if not citations:
        return 0.0
    valid_ids = {
        item.get("evidence_id")
        for item in (verification or {}).get("verified_evidence", [])
    }
    return sum(citation in valid_ids for citation in citations) / len(citations)


def _evidence_sufficiency(answer, verification):
    if not verification or "coverage" not in verification:
        return None
    missing_terms = (verification.get("coverage") or {}).get("missing_terms", [])
    acknowledged = "证据不足" in str(answer or "") or "信息不足" in str(answer or "")
    return 1.0 if bool(missing_terms) == acknowledged else 0.0


def _baseline_metrics(result):
    return {
        "task_completion": _task_completion(result.get("answer"), result.get("status")),
        "evidence_coverage": None,
        "citation_traceability": None,
        "evidence_sufficiency": None,
        "latency": result.get("latency", 0.0),
    }


def _agent_metrics(result):
    verification = result.get("verification") or {}
    coverage = verification.get("coverage") or {}
    return {
        "task_completion": _task_completion(result.get("answer"), result.get("status")),
        "evidence_coverage": coverage.get("coverage_ratio"),
        "citation_traceability": _citation_traceability(
            result.get("answer"),
            verification,
        ),
        "evidence_sufficiency": _evidence_sufficiency(
            result.get("answer"),
            verification,
        ),
        "latency": result.get("latency", 0.0),
    }


def _average(values):
    values = [value for value in values if isinstance(value, (int, float))]
    return sum(values) / len(values) if values else None


def _rate(results, key, expected):
    return sum(item[key] == expected for item in results) / len(results) if results else 0.0


def _metric_average(results, side, metric):
    return _average(item["metrics"][side].get(metric) for item in results)


def _build_overall(results):
    return {
        "total_cases": len(results),
        "baseline_success_rate": _rate(results, "baseline_status", STATUS_SUCCESS),
        "agent_success_rate": _rate(results, "agent_status", STATUS_SUCCESS),
        "baseline_timeout_rate": _rate(results, "baseline_status", STATUS_TIMEOUT),
        "agent_timeout_rate": _rate(results, "agent_status", STATUS_TIMEOUT),
        "average_baseline_latency": _average(
            item["baseline_latency"]
            for item in results
            if item["baseline_status"] != STATUS_TIMEOUT
        ),
        "average_agent_latency": _average(
            item["agent_latency"]
            for item in results
            if item["agent_status"] != STATUS_TIMEOUT
        ),
        "average_agent_evidence_coverage": _metric_average(
            results, "agent", "evidence_coverage"
        ),
        "citation_traceability_rate": _metric_average(
            results, "agent", "citation_traceability"
        ),
        "evidence_sufficiency_rate": _metric_average(
            results, "agent", "evidence_sufficiency"
        ),
    }


def _format_optional(value):
    return "N/A" if value is None else f"{value:.3f}"


def _build_summary(payload):
    results = payload["results"]
    overall = payload["overall"]
    lines = [
        "# Baseline vs Agent Evaluation",
        "",
        "Engineering metrics only; this is not an academic benchmark.",
        "Rates and averages ignore unavailable metrics, while status rates include every case.",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in overall.items():
        lines.append(f"| {key} | {_format_optional(value) if key != 'total_cases' else value} |")

    lines.extend([
        "",
        "## Cases",
        "",
        "| Case | Baseline status | Agent status | Baseline latency | Agent latency | Coverage | Citation traceability | Evidence sufficiency |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for result in results:
        agent_metrics = result["metrics"]["agent"]
        lines.append(
            "| {case_id} | {baseline_status} | {agent_status} | {baseline_latency:.3f} | {agent_latency:.3f} | {coverage} | {citation} | {sufficiency} |".format(
                case_id=result["case_id"],
                baseline_status=result["baseline_status"],
                agent_status=result["agent_status"],
                baseline_latency=result["baseline_latency"],
                agent_latency=result["agent_latency"],
                coverage=_format_optional(agent_metrics["evidence_coverage"]),
                citation=_format_optional(agent_metrics["citation_traceability"]),
                sufficiency=_format_optional(agent_metrics["evidence_sufficiency"]),
            )
        )

    failures = [
        result
        for result in results
        if result["baseline_status"] != STATUS_SUCCESS
        or result["agent_status"] != STATUS_SUCCESS
    ]
    lines.extend(["", "## Failures", ""])
    if not failures:
        lines.append("None")
    else:
        for result in failures:
            if result["baseline_status"] != STATUS_SUCCESS:
                lines.append(
                    f"- {result['case_id']} baseline [{result['baseline_status']}]: {result['baseline_error']}"
                )
            if result["agent_status"] != STATUS_SUCCESS:
                lines.append(
                    f"- {result['case_id']} agent [{result['agent_status']}]: {result['agent_error']}"
                )
    return "\n".join(lines) + "\n"


def run_evaluation(cases, output_dir, case_timeout):
    results = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']}: {case['question']}")
        try:
            run_result = _run_case(case, case_timeout)
        except Exception as error:
            run_result = {
                "baseline": _empty_result(STATUS_EXCEPTION, error=str(error)),
                "agent": _empty_result(STATUS_EXCEPTION, error=str(error)),
            }

        baseline = run_result["baseline"]
        agent = run_result["agent"]
        result = {
            "case_id": case["id"],
            "category": case.get("category"),
            "question": case["question"],
            "baseline_answer": baseline.get("answer", ""),
            "agent_answer": agent.get("answer", ""),
            "baseline_status": baseline.get("status", STATUS_EXCEPTION),
            "agent_status": agent.get("status", STATUS_EXCEPTION),
            "baseline_latency": baseline.get("latency", 0.0),
            "agent_latency": agent.get("latency", 0.0),
            "observations": agent.get("observations", []),
            "verification": agent.get("verification", {}),
            "metrics": {
                "baseline": _baseline_metrics(baseline),
                "agent": _agent_metrics(agent),
            },
            "baseline_error": baseline.get("error"),
            "agent_error": agent.get("error"),
            "agent_events": agent.get("events", []),
        }
        results.append(result)
        print(f"  baseline={result['baseline_status']} agent={result['agent_status']}")

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline_model": BASELINE_MODEL,
        "final_model": FINAL_MODEL,
        "router_model": ROUTER_MODEL,
        "timeout_seconds": case_timeout,
        "working_directory": str(PROJECT_ROOT),
        "metric_note": "Engineering proxies only; not an academic benchmark.",
        "overall": _build_overall(results),
        "results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        _build_summary(payload),
        encoding="utf-8",
    )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("test_cases.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N cases, useful for smoke tests.")
    parser.add_argument("--case-timeout", type=int, default=300, help="Maximum seconds per Baseline or Agent call before recording timeout.")
    args = parser.parse_args()
    cases = _load_cases(args.cases)
    if args.limit is not None:
        cases = cases[:args.limit]
    run_evaluation(cases, args.output_dir, args.case_timeout)


if False:
    main()
"""Run a small, repeatable Baseline vs Agent evaluation.

The metrics in this script are transparent proxies, not academic benchmark scores.
"""

import argparse
import json
import multiprocessing
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "app"
sys.path.insert(0, str(APP_ROOT))

from agent import FINAL_MODEL, run_agent_once  # noqa: E402
from llm import call_llm  # noqa: E402


CITATION_PATTERN = re.compile(r"\[Evidence\s+(\d+)\]")

BASELINE_SYSTEM_PROMPT = """You are a technical analyst answering directly without external tools.
Answer the user's question clearly and honestly. Do not invent sources, URLs,
or Evidence citations. If the question asks for information that is not
provided in the prompt, state uncertainty instead of pretending to have verified it.
Use Markdown when useful."""


def _load_cases(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _safe_call_baseline(question):
    try:
        return {
            "answer": call_llm(
                question,
                model=FINAL_MODEL,
                system_prompt=BASELINE_SYSTEM_PROMPT,
            ),
            "error": None,
        }
    except Exception as error:
        return {"answer": "", "error": str(error)}


def _run_baseline(question):
    started = time.perf_counter()
    result = _safe_call_baseline(question)
    result["latency"] = time.perf_counter() - started
    return result


def _run_agent(question):
    events = []
    started = time.perf_counter()
    try:
        result = run_agent_once(question, event_callback=events.append)
        return {
            "answer": result.get("final_answer", ""),
            "observations": result.get("observations", []),
            "verification": result.get("verification", {}),
            "events": events,
            "error": None,
            "latency": time.perf_counter() - started,
        }
    except Exception as error:
        return {
            "answer": "",
            "observations": [],
            "verification": {},
            "events": events,
            "error": str(error),
            "latency": time.perf_counter() - started,
        }


def _evaluate_call(kind, question, result_queue):
    result = _run_baseline(question) if kind == "baseline" else _run_agent(question)
    result_queue.put(result)


def _run_single_call(kind, question, timeout):
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_evaluate_call,
        args=(kind, question, result_queue),
    )
    started = time.perf_counter()
    process.start()
    process.join(timeout)
    elapsed = time.perf_counter() - started

    if process.is_alive():
        process.terminate()
        process.join()
        failure = f"case exceeded evaluation timeout of {timeout} seconds"
        return {
            "answer": "",
            "observations": [],
            "verification": {},
            "events": [],
            "error": failure,
            "latency": elapsed,
        }

    if not result_queue.empty():
        return result_queue.get()

    failure = f"evaluation process exited with code {process.exitcode}"
    return {
        "answer": "",
        "observations": [],
        "verification": {},
        "events": [],
        "error": failure,
        "latency": elapsed,
    }


def _run_case_with_timeout(case, timeout):
    baseline = _run_single_call("baseline", case["question"], timeout)
    agent = _run_single_call("agent", case["question"], timeout)
    return {"baseline": baseline, "agent": agent}


def _contains_all(text, terms):
    text = str(text or "").casefold()
    return all(str(term).casefold() in text for term in terms)


def _citation_traceability(answer, verification):
    citations = [int(value) for value in CITATION_PATTERN.findall(str(answer or ""))]
    valid_ids = {
        item.get("evidence_id")
        for item in verification.get("verified_evidence", [])
    }
    if not citations:
        return 0.0
    return sum(citation in valid_ids for citation in citations) / len(citations)


def _agent_metrics(case, agent_result):
    answer = agent_result.get("answer", "")
    verification = agent_result.get("verification") or {}
    coverage = verification.get("coverage") or {}
    missing_terms = coverage.get("missing_terms", [])
    evidence = verification.get("verified_evidence", [])
    citation_traceability = _citation_traceability(answer, verification)
    evidence_gap_acknowledged = "证据不足" in answer or "信息不足" in answer
    completed = bool(answer.strip()) and (
        _contains_all(answer, case["expected_terms"])
        or evidence_gap_acknowledged
    )
    grounding = bool(evidence) and citation_traceability == 1.0
    if missing_terms:
        grounding = grounding and evidence_gap_acknowledged
    hallucination = bool(missing_terms) and not evidence_gap_acknowledged
    return {
        "task_completion": 1.0 if completed else 0.0,
        "evidence_coverage": coverage.get("coverage_ratio", 0.0),
        "citation_traceability": citation_traceability,
        "grounding": 1.0 if grounding else 0.0,
        "hallucination": 1.0 if hallucination else 0.0,
        "latency": agent_result.get("latency", 0.0),
    }


def _baseline_metrics(case, baseline_result):
    answer = baseline_result.get("answer", "")
    completed = bool(answer.strip()) and _contains_all(
        answer,
        case["expected_terms"],
    )
    evidence_required_without_evidence = (
        case["requires_evidence"] and bool(answer.strip())
    )
    return {
        "task_completion": 1.0 if completed else 0.0,
        "evidence_coverage": 0.0,
        "citation_traceability": 0.0,
        "grounding": 0.0,
        "hallucination": 1.0 if evidence_required_without_evidence else 0.0,
        "latency": baseline_result.get("latency", 0.0),
    }


def _format_metric(value):
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return str(value)


def _build_summary(results):
    lines = [
        "# Baseline vs Agent Evaluation",
        "",
        "Metrics are transparent heuristic proxies, not academic benchmark scores.",
        "`hallucination` is a risk flag where 1.0 means a detected unsupported-answer risk; lower is better.",
        "",
        "| ID | Category | Baseline completion | Agent completion | Baseline coverage | Agent coverage | Baseline grounding | Agent grounding | Baseline hallucination | Agent hallucination |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        baseline = result["metrics"]["baseline"]
        agent = result["metrics"]["agent"]
        lines.append(
            "| {id} | {category} | {bc} | {ac} | {bv} | {av} | {bg} | {ag} | {bh} | {ah} |".format(
                id=result["id"],
                category=result["category"],
                bc=_format_metric(baseline["task_completion"]),
                ac=_format_metric(agent["task_completion"]),
                bv=_format_metric(baseline["evidence_coverage"]),
                av=_format_metric(agent["evidence_coverage"]),
                bg=_format_metric(baseline["grounding"]),
                ag=_format_metric(agent["grounding"]),
                bh=_format_metric(baseline["hallucination"]),
                ah=_format_metric(agent["hallucination"]),
            )
        )
    return "\n".join(lines) + "\n"


def run_evaluation(cases, output_dir):
    results = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']}: {case['question']}")
        run_result = _run_case_with_timeout(case, run_evaluation.case_timeout)
        baseline = run_result["baseline"]
        agent = run_result["agent"]
        result = {
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "baseline_answer": baseline["answer"],
            "agent_answer": agent["answer"],
            "observations": agent["observations"],
            "verification": agent["verification"],
            "metrics": {
                "baseline": _baseline_metrics(case, baseline),
                "agent": _agent_metrics(case, agent),
            },
            "baseline_error": baseline["error"],
            "agent_error": agent["error"],
            "agent_events": agent["events"],
        }
        results.append(result)
        print(
            "  baseline={baseline} agent={agent}".format(
                baseline="failure" if baseline["error"] else "ok",
                agent="failure" if agent["error"] else "ok",
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "final_model": FINAL_MODEL,
        "metric_note": "Heuristic proxies only; not an academic benchmark.",
        "results": results,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        _build_summary(results),
        encoding="utf-8",
    )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).with_name("test_cases.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N cases, useful for smoke tests.",
    )
    parser.add_argument(
        "--case-timeout",
        type=int,
        default=300,
        help="Maximum seconds per Baseline or Agent call before recording failure.",
    )
    args = parser.parse_args()
    cases = _load_cases(args.cases)
    if args.limit is not None:
        cases = cases[:args.limit]
    run_evaluation.case_timeout = args.case_timeout
    run_evaluation(cases, args.output_dir)


if False:
    main()
'''
