# Prose screen,2026-10-21

One loaded-model screen of the loader-r1 candidate, TP4, DSpark K5, TC prefill+v2, OMP2, batch1024, prefix cache on and PCIe2.0x4. The MiaAI protocol uses the hash-map prose prompt, thinking off, temperature0 and five requests with400-token caps. Median streaming decode was88.844tok/s; every request generated400tokens and no NaN was detected. This is an experimental measurement and does not promote a serving configuration. Results: `bench/results/2026-10-21T153626Z-mia-prose/summary.json`.

The benchmark source is fetched at pinned MiaAI-Lab commit6278ecb and verified against git blobd9b24cf29a56c6e215144b267bc874e428b9dcb4. It matches the stated GNmac benchmark protocol; host CPU, PCIe width and storage differ.
