from __future__ import annotations

import argparse
import os
from multiprocessing import get_context
import numpy as np



def build_transition_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nxt = np.empty((4096, 2, 2), dtype=np.uint8)
    wrt = np.empty((4096, 2, 2), dtype=np.uint8)
    mov = np.empty((4096, 2, 2), dtype=np.int8)

    configs = ((0, 1), (0, 0), (1, 1), (1, 0))

    for r in range(4096):
        d0 = (r >> 9) & 7
        d1 = (r >> 6) & 7
        d2 = (r >> 3) & 7
        d3 = r & 7
        for digit, (s, sym) in zip((d0, d1, d2, d3), configs):
            nxt[r, s, sym] = (digit >> 2) & 1
            wrt[r, s, sym] = (digit >> 1) & 1
            mov[r, s, sym] = -1 if (digit & 1) == 0 else 1

    return nxt, wrt, mov



def build_drift_left_on_blank(nxt: np.ndarray, mov: np.ndarray) -> np.ndarray:
    drift = np.zeros((4096, 2), dtype=np.uint8)
    for r in range(4096):
        for s0 in (0, 1):
            s = s0
            ok = True
            for _ in range(3):
                if mov[r, s, 0] != -1:
                    ok = False
                    break
                s = int(nxt[r, s, 0])
            drift[r, s0] = 1 if ok else 0
    return drift



def build_bitlen_table(max_n: int) -> np.ndarray:
    cnt = 1 << max_n
    bitlen = np.empty(cnt, dtype=np.uint8)
    for x in range(cnt):
        bl = x.bit_length()
        bitlen[x] = 1 if bl == 0 else bl
    return bitlen



NXT, WRT, MOV = build_transition_tables()
DRIFT = build_drift_left_on_blank(NXT, MOV)

HAVE_NUMBA = False
_NUMBA_IMPORT_ERROR: Exception | None = None
try:
    from numba import njit
    HAVE_NUMBA = True
except Exception as e:
    _NUMBA_IMPORT_ERROR = e



if HAVE_NUMBA:
    @njit(cache=True, inline="always")
    def _simulate_one(
        rule: int,
        inp: np.uint32,
        D: int,
        seen: np.ndarray,      
        tape_val: np.ndarray,
        tape_stamp: np.ndarray,
        drift: np.ndarray,
        nxt: np.ndarray,
        wrt: np.ndarray,
        mov: np.ndarray,
        run_id: np.uint32,
        step_cap: int,
    ) -> int:
        state = 0
        depth = 0

        small_ok = True
        mask_limit = (np.uint32(1) << (D + 1)) - np.uint32(1)
        mask = inp & mask_limit

        idx0 = (mask * (D + 1) + depth) * 2 + state
        seen[idx0] = run_id

        for step in range(1, step_cap + 1):
            if depth > D:
                small_ok = False

            if depth <= D:
                sym = (mask >> depth) & np.uint32(1)
            else:
                if depth >= tape_val.shape[0]:
                    return 0
                fresh = tape_stamp[depth] != run_id
                if fresh:
                    if depth < 32:
                        sym = (inp >> depth) & np.uint32(1)
                    else:
                        sym = np.uint32(0)
                    tape_val[depth] = np.uint8(sym)
                    tape_stamp[depth] = run_id

                    if sym == 0 and drift[rule, state] == 1:
                        return 0
                else:
                    sym = np.uint32(tape_val[depth])

            ns = nxt[rule, state, sym]
            w = wrt[rule, state, sym]
            delta = mov[rule, state, sym]

            if depth <= D:
                bit = np.uint32(1) << depth
                if w == 1:
                    mask |= bit
                else:
                    mask &= ~bit
            else:
                tape_val[depth] = w

            state = ns
            depth -= int(delta)

            if depth < 0:
                return step

            if small_ok and depth <= D:
                idx = (mask * (D + 1) + depth) * 2 + state
                if seen[idx] == run_id:
                    return 0
                seen[idx] = run_id

        return 0

    @njit(cache=True)
    def _process_rule_chunk(
        rule_start: int,
        rule_end: int,
        max_n: int,
        bitlen: np.ndarray,
        D: int,
        step_cap: int,
        drift: np.ndarray,
        nxt: np.ndarray,
        wrt: np.ndarray,
        mov: np.ndarray,
    ) -> np.ndarray:
        input_count = 1 << max_n
        max_exact = np.zeros(max_n + 1, dtype=np.int32)

        seen_size = (1 << (D + 1)) * (D + 1) * 2
        seen = np.zeros(seen_size, dtype=np.uint32)

        tape_len = max(step_cap + 2, D + 2, 64)
        tape_val = np.zeros(tape_len, dtype=np.uint8)
        tape_stamp = np.zeros(tape_len, dtype=np.uint32)

        run_id = np.uint32(1)

        for rule in range(rule_start, rule_end):
            for x in range(input_count):
                t = _simulate_one(
                    rule=rule,
                    inp=np.uint32(x),
                    D=D,
                    seen=seen,
                    tape_val=tape_val,
                    tape_stamp=tape_stamp,
                    drift=drift,
                    nxt=nxt,
                    wrt=wrt,
                    mov=mov,
                    run_id=run_id,
                    step_cap=step_cap,
                )
                if t > 0:
                    bl = int(bitlen[x])
                    if t > max_exact[bl]:
                        max_exact[bl] = t
                run_id += np.uint32(1)

        return max_exact



def _worker_task(args: tuple[int, int, int, int, int]) -> np.ndarray:
    rule_start, rule_end, max_n, D, step_cap = args

    bitlen = build_bitlen_table(max_n)

    if not HAVE_NUMBA:
        raise RuntimeError(
            "Numba is required for the optimized path, but import failed:\n"
            f"{_NUMBA_IMPORT_ERROR}"
        )

    return _process_rule_chunk(
        rule_start=rule_start,
        rule_end=rule_end,
        max_n=max_n,
        bitlen=bitlen,
        D=D,
        step_cap=step_cap,
        drift=DRIFT,
        nxt=NXT,
        wrt=WRT,
        mov=MOV,
    )



def _split_rules(num_workers: int) -> list[tuple[int, int]]:
    rules = 4096
    chunk = (rules + num_workers - 1) // num_workers
    out = []
    for i in range(num_workers):
        a = i * chunk
        b = min(rules, a + chunk)
        if a < b:
            out.append((a, b))
    return out



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=12, help="Compute maxima for n=1..max-n (default 12).")
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes (default: physical/core count heuristic).",
    )
    ap.add_argument(
        "--D",
        type=int,
        default=16,
        help="Exact cycle-detection depth cutoff D (default 16). Must be <= 30.",
    )
    ap.add_argument(
        "--step-cap",
        type=int,
        default=1_000_000,
        help="Max steps per (rule,input) run after leaving depth<=D (default 1,000,000).",
    )
    args = ap.parse_args()

    max_n = int(args.max_n)
    if max_n < 1 or max_n > 24:
        raise SystemExit("Choose --max-n in [1..24]. (Beyond that, input count explodes.)")

    D = int(args.D)
    if D < 0 or D > 30:
        raise SystemExit("--D must be in [0..30]. (Mask uses 32-bit ops.)")

    step_cap = int(args.step_cap)
    if step_cap < 1:
        raise SystemExit("--step-cap must be >= 1")

    if args.workers and args.workers > 0:
        workers = int(args.workers)
    else:
        try:
            workers = len(os.sched_getaffinity(0))
        except Exception:
            workers = os.cpu_count() or 1
        workers = min(workers, 32)

    chunks = _split_rules(workers)
    tasks = [(a, b, max_n, D, step_cap) for (a, b) in chunks]

    ctx = get_context("spawn")
    with ctx.Pool(processes=len(tasks)) as pool:
        parts = pool.map(_worker_task, tasks)

    max_exact = np.zeros(max_n + 1, dtype=np.int32)
    for arr in parts:
        max_exact = np.maximum(max_exact, arr)

    prefix = []
    cur = 0
    for n in range(1, max_n + 1):
        if int(max_exact[n]) > cur:
            cur = int(max_exact[n])
        prefix.append(cur)

    print(prefix)


"""
Usage:
    python tm22_max_halting.py --max-n 6 --workers 32
"""



if __name__ == "__main__":
    main()
