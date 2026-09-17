"""Stable Baseline vs Agent evaluation implementation."""

import json
import multiprocessing
import os
import queue
import re
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "app"
import sys
sys.path.insert(0, str(APP_ROOT))

from agent import FINAL_MODEL, ROUTER_MODEL, run_agent_once
from llm import call_llm

BASELINE_MODEL = FINAL_MODEL
CITATION_PATTERN = re.compile(r"\[Evidence\s+(\d+)\]")
SUCCESS = "success"
TIMEOUT = "timeout"
TOOL_FAILURE = "tool_failure"
EXCEPTION = "exception"
FAILURE_STATUSES = {TIMEOUT, TOOL_FAILURE, EXCEPTION}

BASELINE_SYSTEM_PROMPT = """You are a technical analyst answering directly without external tools.
Answer clearly and honestly. Do not invent sources, URLs, or Evidence citations.
Use Markdown when useful."""


def empty_result(status, latency=0.0, error=None, events=None):
    events = list(events or [])
    return {"answer": "", "observations": [], "verification": {}, "events": events,
            "last_event": events[-1] if events else None,
            "status": status, "error": error, "latency": latency}


def run_baseline(question):
    started = time.perf_counter()
    try:
        answer = call_llm(question, model=BASELINE_MODEL, system_prompt=BASELINE_SYSTEM_PROMPT)
        result = {"answer": answer, "status": SUCCESS, "error": None}
    except Exception as error:
        result = {"answer": "", "status": EXCEPTION, "error": str(error)}
    result["latency"] = time.perf_counter() - started
    return result


def run_agent(question, event_queue=None):
    events = []
    started = time.perf_counter()

    def record_event(event):
        payload = dict(event)
        payload["elapsed"] = time.perf_counter() - started
        events.append(payload)
        if event_queue is not None:
            event_queue.put(payload)

    try:
        result = run_agent_once(question, event_callback=record_event)
        observations = result.get("observations", [])
        failures = []
        for observation in observations:
            tool_result = observation.get("tool_result") or {}
            if tool_result.get("success") is not True:
                failures.append(
                    f"{tool_result.get('source', observation.get('action', 'unknown'))}: "
                    f"{tool_result.get('error', 'unsuccessful tool result')}"
                )
        return {
            "answer": result.get("final_answer", ""),
            "observations": observations,
            "verification": result.get("verification", {}),
            "events": events,
            "last_event": events[-1] if events else None,
            "status": result.get("status", TOOL_FAILURE if failures else SUCCESS),
            "error": result.get("error") or ("; ".join(failures) if failures else None),
            "latency": time.perf_counter() - started,
        }
    except Exception as error:
        result = empty_result(EXCEPTION, time.perf_counter() - started, str(error))
        result["events"] = events
        result["last_event"] = events[-1] if events else None
        return result


def worker(kind, question, result_queue, event_queue):
    os.chdir(PROJECT_ROOT)
    result_queue.put(
        run_baseline(question)
        if kind == "baseline"
        else run_agent(question, event_queue)
    )


def drain_events(event_queue, events):
    while True:
        try:
            events.append(event_queue.get_nowait())
        except queue.Empty:
            return


def run_call(kind, question, timeout, worker_target=None):
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    event_queue = context.Queue()
    process = context.Process(
        target=worker_target or worker,
        args=(kind, question, result_queue, event_queue),
    )
    started = time.perf_counter()
    events = []
    result = None

    def close_queues():
        drain_events(event_queue, events)
        result_queue.close()
        result_queue.join_thread()
        event_queue.close()
        event_queue.join_thread()

    try:
        process.start()
        deadline = started + timeout
        while True:
            drain_events(event_queue, events)
            try:
                result = result_queue.get_nowait()
                break
            except queue.Empty:
                pass

            if not process.is_alive():
                break

            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            process.join(min(0.1, remaining))
    except Exception as error:
        if process.is_alive():
            process.terminate()
            process.join()
        drain_events(event_queue, events)
        close_queues()
        return empty_result(EXCEPTION, time.perf_counter() - started, str(error), events)

    elapsed = time.perf_counter() - started
    if result is None and not process.is_alive():
        try:
            result = result_queue.get(timeout=1)
        except queue.Empty:
            result = None

    if result is None and process.is_alive():
        process.terminate()
        process.join()
        close_queues()
        return empty_result(
            TIMEOUT,
            elapsed,
            f"evaluation timeout after {timeout} seconds",
            events,
        )
    if result is None:
        close_queues()
        return empty_result(
            EXCEPTION,
            elapsed,
            f"worker exited with code {process.exitcode} without a result",
            events,
        )
    if process.is_alive():
        process.terminate()
        process.join()
    close_queues()
    result["latency"] = elapsed
    if events:
        result["events"] = events
        result["last_event"] = events[-1]
    else:
        result.setdefault("events", [])
        result.setdefault("last_event", None)
    return result


def task_completion(result):
    if result.get("status") in FAILURE_STATUSES and not str(result.get("answer", "")).strip():
        return None
    return 1.0 if str(result.get("answer", "")).strip() else 0.0


def citation_traceability(answer, verification):
    citations = [int(value) for value in CITATION_PATTERN.findall(str(answer or ""))]
    if not citations:
        return 0.0
    valid_ids = {item.get("evidence_id") for item in (verification or {}).get("verified_evidence", [])}
    return sum(citation in valid_ids for citation in citations) / len(citations)


def evidence_sufficiency(answer, verification):
    if "coverage" not in (verification or {}):
        return None
    missing = (verification.get("coverage") or {}).get("missing_terms", [])
    acknowledged = "证据不足" in str(answer or "") or "信息不足" in str(answer or "")
    return 1.0 if bool(missing) == acknowledged else 0.0


def metrics(result, agent=False):
    verification = result.get("verification") or {}
    coverage = verification.get("coverage") or {}
    unavailable = (
        result.get("status") in FAILURE_STATUSES
        and not str(result.get("answer", "")).strip()
    )
    return {
        "task_completion": task_completion(result),
        "evidence_coverage": coverage.get("coverage_ratio") if agent and not unavailable else None,
        "citation_traceability": citation_traceability(result.get("answer"), verification) if agent and not unavailable else None,
        "evidence_sufficiency": evidence_sufficiency(result.get("answer"), verification) if agent and not unavailable else None,
        "latency": result.get("latency", 0.0),
    }


def average(values):
    values = [value for value in values if isinstance(value, (int, float))]
    return sum(values) / len(values) if values else None


def rate(results, field, value):
    return sum(result[field] == value for result in results) / len(results) if results else 0.0


def overall(results):
    return {
        "total_cases": len(results),
        "baseline_success_rate": rate(results, "baseline_status", SUCCESS),
        "agent_success_rate": rate(results, "agent_status", SUCCESS),
        "baseline_timeout_rate": rate(results, "baseline_status", TIMEOUT),
        "agent_timeout_rate": rate(results, "agent_status", TIMEOUT),
        "average_baseline_latency": average(result["baseline_latency"] for result in results if result["baseline_status"] != TIMEOUT),
        "average_agent_latency": average(result["agent_latency"] for result in results if result["agent_status"] != TIMEOUT),
        "average_agent_evidence_coverage": average(result["metrics"]["agent"].get("evidence_coverage") for result in results),
        "citation_traceability_rate": average(result["metrics"]["agent"].get("citation_traceability") for result in results),
        "evidence_sufficiency_rate": average(result["metrics"]["agent"].get("evidence_sufficiency") for result in results),
    }


def fmt(value):
    return "N/A" if value is None else f"{value:.3f}"


def build_summary(payload):
    lines = ["# Baseline vs Agent Evaluation", "", "Engineering metrics only; not an academic benchmark.", "Unavailable metrics are shown as N/A.", "", "## Overall", "", "| Metric | Value |", "|---|---:|"]
    for key, value in payload["overall"].items():
        lines.append(f"| {key} | {value if key == 'total_cases' else fmt(value)} |")
    lines.extend(["", "## Cases", "", "| Case | Baseline status | Agent status | Baseline latency | Agent latency | Coverage | Citation traceability | Evidence sufficiency |", "|---|---|---|---:|---:|---:|---:|---:|"])
    for result in payload["results"]:
        agent_metrics = result["metrics"]["agent"]
        lines.append(f"| {result['case_id']} | {result['baseline_status']} | {result['agent_status']} | {result['baseline_latency']:.3f} | {result['agent_latency']:.3f} | {fmt(agent_metrics['evidence_coverage'])} | {fmt(agent_metrics['citation_traceability'])} | {fmt(agent_metrics['evidence_sufficiency'])} |")
    lines.extend(["", "## Failures", ""])
    failures = [result for result in payload["results"] if result["baseline_status"] != SUCCESS or result["agent_status"] != SUCCESS]
    if not failures:
        lines.append("None")
    for result in failures:
        if result["baseline_status"] != SUCCESS:
            lines.append(f"- {result['case_id']} baseline [{result['baseline_status']}]: {result['baseline_error']}")
        if result["agent_status"] != SUCCESS:
            lines.append(f"- {result['case_id']} agent [{result['agent_status']}]: {result['agent_error']}")
    return "\n".join(lines) + "\n"


def run_evaluation(cases, output_dir, timeout):
    results = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case['id']}: {case['question']}", flush=True)
        try:
            baseline = run_call("baseline", case["question"], timeout)
            agent = run_call("agent", case["question"], timeout)
        except Exception as error:
            baseline = empty_result(EXCEPTION, error=str(error))
            agent = empty_result(EXCEPTION, error=str(error))
        results.append({
            "case_id": case["id"],
            "category": case.get("category"),
            "question": case["question"],
            "baseline_answer": baseline.get("answer", ""),
            "agent_answer": agent.get("answer", ""),
            "baseline_status": baseline.get("status", EXCEPTION),
            "agent_status": agent.get("status", EXCEPTION),
            "baseline_latency": baseline.get("latency", 0.0),
            "agent_latency": agent.get("latency", 0.0),
            "observations": agent.get("observations", []),
            "verification": agent.get("verification", {}),
            "metrics": {"baseline": metrics(baseline), "agent": metrics(agent, agent=True)},
            "baseline_error": baseline.get("error"),
            "agent_error": agent.get("error"),
            "agent_events": agent.get("events", []),
            "agent_last_event": agent.get("last_event"),
        })
        print(f"  baseline={baseline.get('status')} agent={agent.get('status')}", flush=True)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline_model": BASELINE_MODEL,
        "final_model": FINAL_MODEL,
        "router_model": ROUTER_MODEL,
        "timeout_seconds": timeout,
        "working_directory": str(PROJECT_ROOT),
        "metric_note": "Engineering proxies only; not an academic benchmark.",
        "overall": overall(results),
        "results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("test_cases.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-timeout", type=int, default=300)
    args = parser.parse_args()
    with args.cases.open("r", encoding="utf-8") as file:
        cases = json.load(file)
    if args.limit is not None:
        cases = cases[:args.limit]
    run_evaluation(cases, args.output_dir, args.case_timeout)


if __name__ == "__main__":
    main()
