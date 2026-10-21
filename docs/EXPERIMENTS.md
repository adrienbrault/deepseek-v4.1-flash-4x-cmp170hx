# Experiment diagnostics

The telemetry collector records GPU state, CPU frequencies and faults, disk activity, process CPU and server metrics. The unscored diagnostic helper captures short c1 and c4-with-prefill Torch CPU/CUDA traces after benchmark scores are finalized, plus native CPU samples. Enable the profiler at model boot and pass a result directory to `run-experiment-diagnostics.sh`. A confirmed profiler stop is required before scored work resumes; missing or unsafe receipts fail the wrapper. Trace summaries report summed kernel durations and communication fractions, which are not wall-time percentages or device utilization.

The first hardware capture completed on2026-10-21 with all four ranks and eight traces, without capture errors. Subsequent helpers also target EngineCore and API processes in native CPU sampling. Five diagnostic lifecycle tests, four telemetry tests and four summary tests passed on Linux; synthetic tests do not replace a hardware capture.

The pinned prose result is in [the measurement notes](../bench/results/2026-10-21T153626Z-mia-prose/NOTES.md). No daily serving configuration is promoted by these instruments.
