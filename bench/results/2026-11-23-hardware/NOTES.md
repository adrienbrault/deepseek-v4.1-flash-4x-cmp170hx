# Hardware and memory of the served boot, 2026-11-23 UTC

Recorded on the 4x CMP 170HX box on 2026-11-23 between 07:44 and 08:00 UTC, with `current-r18b` serving (boot of 07:44 UTC, all four cards at PCIe Gen2 x16, 175 W cap per card). Read-only measurements; the server was idle.

## Host

- CPU: AMD EPYC 7371, 16 cores, SMT off (16 logical CPUs). It replaced the EPYC 7401P on 2026-11-02; results before that date name the 7401P.
- RAM: 64 GB, eight 8 GB DDR4-2667 registered DIMMs, one per channel.
- Board: GIGABYTE MZ31-AR0.
- GPU driver 615.71.09, Linux 7.0.0-43. Maximum clocks reported by the driver: memory 1,728 MHz, SM 1,845 MHz.
- Model storage: two Samsung 970 EVO Plus 2 TB SSDs in RAID0 (512 KiB chunks, ext4). The checkpoint and the two Engram tables are read from this array.

## Engram read rate

`fio` sequential read of one Engram table file (`--rw=read --bs=1M --iodepth=32 --ioengine=libaio --direct=1 --readonly`, 12 GiB from a 40 GiB offset): 6,345,391 KiB/s, which is 6.5 GB/s. One run.

## Memory on the served boot

Per card, from the vLLM startup log of the 07:44 UTC boot (rank 0; ranks 1-3 within 0.1 GiB): model weights 53.6 GiB, KV cache 5.25 GiB, CUDA graphs 0.27 + 0.56 GiB, peak activations 1.29 GiB. The engine reported a KV pool of 2,483,207 tokens (9.47 requests at the full 262,144-token window). `nvidia-smi` reported 64,126 to 64,268 MiB used of 65,536 MiB per card (62.6 to 62.8 GiB).

Totals over the four cards: weights 214.3 GiB, KV cache 21.0 GiB, everything else (graphs, activations, CUDA context, allocator reserve) about 15.5 GiB, out of 256 GiB.

Host memory: the serving container used 25.5 GiB of its 52 GiB limit (`docker stats`, idle server).
