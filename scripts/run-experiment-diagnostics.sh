#!/usr/bin/env bash
# The benchmark receipt is already finalized. Diagnostic failures stay explicit.
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd)
R=${1:?results root}
tag=${2:?arm tag}
A=$R/$tag
if [[ -f $A/diagnostics/capture-receipt.json ]]; then
    echo "DIAGNOSTIC_REUSE $tag existing receipt; no duplicate capture"
    python3 - "$A/diagnostics/capture-receipt.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get('safe_for_scored_work',False) else 3)
PY
    exit "$?"
fi
mkdir -p "$A/diagnostics"
echo "DIAGNOSTIC_START $tag $(date -u +%FT%TZ) unscored"
diagnostic_rc=0
timeout --kill-after=100 180 python3 "$H/experiment-diagnostics.py" "$A/diagnostics" \
    --prompt-doc "$R/prompt-doc.txt" > "$A/diagnostics/run.log" 2>&1 || diagnostic_rc=$?
summary_rc=0
python3 "$H/summarize-profiler.py" "$A/torch-profiles" \
    --output "$A/diagnostics/gpu-summary.json" > "$A/diagnostics/summary.log" 2>&1 || summary_rc=$?
python3 - "$A/diagnostics" "$diagnostic_rc" "$summary_rc" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); diagnostic_rc,summary_rc=map(int,sys.argv[2:])
meta=json.loads((p/'diagnostics.json').read_text()) if (p/'diagnostics.json').exists() else {}
result={'status':'PASS' if diagnostic_rc==summary_rc==0 and meta.get('status')=='completed' else 'PARTIAL',
        'classification':'unscored diagnostic; does not alter benchmark receipt',
        'diagnostic_exit':diagnostic_rc,'trace_summary_exit':summary_rc,
        'cpu_gpu_capture_status':meta.get('status','missing'),
        'safe_for_scored_work':meta.get('safe_for_scored_work',False)}
(p/'capture-receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
raise SystemExit(0 if result['safe_for_scored_work'] else 3)
PY
echo "DIAGNOSTIC_END $tag $(date -u +%FT%TZ)"
