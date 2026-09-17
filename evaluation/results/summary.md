# Baseline vs Agent Evaluation

Engineering metrics only; not an academic benchmark.
Unavailable metrics are shown as N/A.

## Overall

| Metric | Value |
|---|---:|
| total_cases | 6 |
| baseline_success_rate | 0.833 |
| agent_success_rate | 0.833 |
| baseline_timeout_rate | 0.000 |
| agent_timeout_rate | 0.167 |
| average_baseline_latency | 88.567 |
| average_agent_latency | 121.236 |
| average_agent_evidence_coverage | 0.200 |
| citation_traceability_rate | 0.800 |
| evidence_sufficiency_rate | 0.200 |

## Cases

| Case | Baseline status | Agent status | Baseline latency | Agent latency | Coverage | Citation traceability | Evidence sufficiency |
|---|---|---|---:|---:|---:|---:|---:|
| T1 | exception | success | 15.964 | 113.469 | 0.000 | 1.000 | 0.000 |
| T2 | success | success | 94.659 | 101.629 | 0.000 | 0.000 | 1.000 |
| T3 | success | success | 149.099 | 104.697 | 0.500 | 1.000 | 0.000 |
| T4 | success | timeout | 90.158 | 177.790 | N/A | N/A | N/A |
| T5 | success | success | 102.918 | 147.873 | 0.500 | 1.000 | 0.000 |
| T6 | success | success | 78.608 | 138.512 | 0.000 | 1.000 | 0.000 |

## Failures

- T1 baseline [exception]: llama-server process has terminated: exit status 0xc0000409: The system detected an overrun of a stack-based buffer in this application. This overrun could potentially allow a malicious user to gain control of this application.: CUDA error: shared object initialization failed
CUDA error (status code: 500)
- T4 agent [timeout]: final LLM call timed out after 150s (model=qwen3:8b)
