#!/usr/bin/env bash
# Collector lifecycle: terminating the observer must reap every owned child.
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
mkdir "$T/bin" "$T/out"
export TEST_COLLECTOR_PIDS="$T/pids"
for command in iostat pidstat nvidia-smi; do
    cat > "$T/bin/$command" <<'MOCK'
#!/usr/bin/env bash
echo "$$" >> "$TEST_COLLECTOR_PIDS"
exec sleep 60
MOCK
    chmod +x "$T/bin/$command"
done
PATH="$T/bin:$PATH" bash "$H/boot-telemetry.sh" "$T/out" & observer=$!
for i in {1..30}; do
    [[ -f $T/pids && $(wc -l < "$T/pids") == 3 ]] && break
    sleep 0.1
done
kill -TERM "$observer"
wait "$observer"
[[ $(wc -l < "$T/pids") == 3 ]]
while read -r pid; do ! kill -0 "$pid" 2>/dev/null; done < "$T/pids"
[[ $(wc -l < "$T/out/boot-telemetry-times.txt") == 2 ]]
echo 'PASS: all telemetry children stopped and reaped'
