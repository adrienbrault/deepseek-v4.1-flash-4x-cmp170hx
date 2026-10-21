# Experimental full matrix,2026-10-21

Results directory `2026-10-21T182229Z-ring-control`. TP4, four CMP170HX cards, accepted TC-MoE V2, OMP2, batch2048. Same immutable ring-capable image, ring disabled, original single-SSD Engram. Worker affinities remained at the control configuration. No daily promotion.

Fixed decode uses400output tokens, ignored EOS, temperature0, seed15, one warmup and three measured batches at c1/c4/c8. Aggregate is the measured sum of stream decode rates. Agent is a growing mixed prefill/decode workload, with observed concurrency in `agent.json`. Prefill uses fresh random-leading approximately2k/8k inputs, one output token and thinking off; all cached tokens explicitly zero. Input rate is total batch input tokens over submit-to-last-first-output, including HTTP, tokenization, scheduling, queueing and first decode. Single batch per prefill point; sequential screens without a confidence interval.
