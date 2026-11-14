# Current serving operation profile, 2026-11-14 UTC

Source: `rounds/profile-r7b-r1/REPORT-20261114T111118Z.md`, results directory `rounds/profile-r7b-r1`. This is a PARTIAL report from one current-r7b profiling boot at c4 K3, c8 K2 and c1 K5. `summary.json` derives only the rank-zero GPU operation shares, rounded to one decimal place. It does not copy traces, memory addresses, model inputs or token data.

Routed MoE accounts for 34.8 / 35.6 / 28.7 % of the step at c4 / c8 / c1; ring all-reduce for 26.6 / 29.6 / 20.3 %, drafter 4.9 / 4.7 / 8.3 %, mHC 4.8 / 4.2 / 5.3 % and attention 2.8 / 2.6 / 3.5 %. Some operations are unassigned or missing, so these shares are not a complete partition. The report identifies no exposed host gap. These measurements have different forced draft depths from the qualification's production K5 configuration and do not replace its speed table. Output length and sampler are not established by the retained report alone.
