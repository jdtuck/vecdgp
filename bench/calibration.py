"""Timing for calibration MCMC: one emulator draw per outer iteration.

A calibration sampler evaluates the emulator once per iteration, at a single
proposed parameter, for a hundred thousand iterations or more. That is the
opposite shape from what ``post_sample`` was built for, so this benchmark
measures the thing that actually runs in the loop -- per-evaluation latency --
rather than throughput over a dense test grid.

Three ways to get one sample at one location are compared:

  post_sample()             every retained draw, then keep one   (the naive way)
  post_sample(draws=t)      only the draw you wanted
  fit.sampler().sample()    per-draw state cached across calls

Two counts are then swept, because "how long does one sample take" has two
different answers depending on which knob you turn:

  nper    samples per call, at one posterior draw -- the per-call setup is
          paid once however many you ask for, so cost per sample falls
  draws   how many posterior draws you touch -- one sample from each, which
          is what a calibration loop actually does, and the cache has to
          hold one state per draw visited

It also runs a real random-walk Metropolis calibration, because microbenchmarks
flatter a cache: the loop visits draws in a scattered order and interleaves
other work, which is where a cache either holds up or does not.

Run::

    python bench/calibration.py                 # n=6000, d=2, m=25
    python bench/calibration.py 20000 2 25      # your own size
    python bench/calibration.py 6000 2 25 --quick
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from vecdgp import fit_two_layer  # noqa: E402

QUICK = "--quick" in sys.argv
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
N = int(ARGS[0]) if len(ARGS) > 0 else 6000
D = int(ARGS[1]) if len(ARGS) > 1 else 2
M = int(ARGS[2]) if len(ARGS) > 2 else 25
NMCMC = 400 if QUICK else 1000
KEEP_BURN, KEEP_THIN = NMCMC // 2, 5
OUTER = 2000 if QUICK else 5000
PROJECT = 100_000  # iteration count to project wall time for


def hms(s):
    if s < 60:
        return f"{s:.0f} s"
    if s < 3600:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"


def timeit(fn, reps, warmup=1):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def simulator(x, theta):
    """Stand-in computer model: output depends on inputs and a calibration par."""
    return np.sin(6 * x[:, 0] * theta[0]) + 0.5 * np.cos(4 * x[:, 1] + theta[1])


def main():
    try:
        import numba

        env = f"numba {numba.__version__}, {numba.get_num_threads()} threads"
    except Exception:
        env = "numba NOT INSTALLED (this will be very slow)"
    print(f"vecdgp calibration benchmark -- {env}")
    print(f"n = {N}, d = {D}, m = {M}\n")

    rng = np.random.default_rng(0)
    x = rng.random((N, D))
    theta_true = np.array([1.0, 0.5])
    y = simulator(x, theta_true)
    y = (y - y.mean()) / y.std()

    t0 = time.time()
    fit = fit_two_layer(x, y, nmcmc=NMCMC, true_g=1e-4, m=M, verb=False, seed=1)
    fit = fit.trim(KEEP_BURN, KEEP_THIN)
    print(f"fit: {NMCMC} sweeps in {hms(time.time() - t0)}, "
          f"{fit.nmcmc} draws retained\n")

    q = rng.random((1, D))
    emu = fit.sampler()

    # ---- do the three routes agree? ---------------------------------------
    # A faster answer is only worth having if it is the same answer.
    print("agreement (the fast paths must not be a different calculation)")
    print("-" * 68)
    t = 7
    mu, sd = emu.mean_sd(q, draw=t)
    ref = fit.predict(q, draws=t, cores=1)
    print(f"  sampler.mean_sd vs predict(draws={t}):  "
          f"|mu| {abs(mu[0] - ref.mean[0]):.2e}   "
          f"|sd| {abs(sd[0] - np.sqrt(ref.s2[0])):.2e}")
    nd = 20000
    a = emu.sample(q, draw=t, rng=np.random.default_rng(1), nper=nd)
    b = fit.post_sample(q, draws=t, nper=nd, rng=np.random.default_rng(2), cores=1)
    se = sd[0] / np.sqrt(nd)
    print(f"  {nd} draws each, sampler vs post_sample:  "
          f"mean {a.mean():+.4f} / {b.mean():+.4f}  (±{2*se:.4f})   "
          f"sd {a.std():.4f} / {b.std():.4f}")

    # ---- per-evaluation latency -------------------------------------------
    print("\nper emulator evaluation: 1 location, fresh posterior draw each step")
    print("-" * 68)
    print(f"  {'method':<34}{'us/eval':>10}{f'x{PROJECT//1000}k':>12}{'speedup':>10}")
    R = np.random.default_rng(9)
    rows = []
    rows.append(("post_sample()  [all draws]",
                 timeit(lambda: fit.post_sample(q, cores=1, rng=R), 3)))
    rows.append(("post_sample(draws=t)",
                 timeit(lambda: fit.post_sample(
                     q, draws=int(R.integers(fit.nmcmc)), cores=1, rng=R), 20)))
    for t_ in range(fit.nmcmc):  # warm every draw's cache
        emu.sample(q, draw=t_)
    rows.append(("sampler().sample(draw=t)",
                 timeit(lambda: emu.sample(
                     q, draw=int(R.integers(fit.nmcmc)), rng=R), 300)))
    base = rows[0][1]
    for name, el in rows:
        print(f"  {name:<34}{el*1e6:9.1f}{hms(el*PROJECT):>12}{base/el:9.0f}x")

    # ---- scaling in the number of samples per call -------------------------
    # The setup (mapping through the latent layer, querying the tree, building
    # the conditioning set) is paid once per call whatever nper is; only the
    # sequential draw itself repeats.  So cost per sample should fall towards
    # a floor as nper grows -- and that floor says what the draw alone costs.
    print("\nsamples per call at one draw: nper amortises the per-call setup")
    print("-" * 68)
    print(f"  {'nper':>7}{'ms/call':>12}{'us/sample':>12}{'vs nper=1':>12}")
    per1 = None
    for nper in (1, 10, 100, 1000, 10000):
        reps = max(3, min(200, 2000 // nper))
        el = timeit(lambda: emu.sample(q, draw=t, rng=R, nper=nper), reps)
        per = el / nper
        per1 = per if per1 is None else per1
        print(f"  {nper:7d}{el*1e3:11.3f}{per*1e6:12.2f}{per1/per:11.1f}x")
    setup = rows[-1][1]
    print(f"  per-call setup ~{setup*1e6:.0f} us, marginal draw ~{per*1e6:.2f} us")

    # ---- scaling in the number of posterior draws visited -------------------
    # The shape a calibration loop actually has: one sample, but from a
    # different posterior iteration each step. Each new draw costs a KD-tree
    # build the first time it is seen, then nothing.
    print("\nposterior draws visited: 1 sample each, new draw every step")
    print("-" * 68)
    print(f"  {'draws':>7}{'us/eval cold':>15}{'us/eval warm':>15}{'x100k warm':>12}")
    grid = sorted({v for v in (1, 5, 25, 100, fit.nmcmc) if v <= fit.nmcmc})
    for nd_ in grid:
        picks = np.linspace(0, fit.nmcmc - 1, nd_).astype(int)
        cold_emu = fit.sampler()
        t0 = time.perf_counter()
        for tt in picks:
            cold_emu.sample(q, draw=int(tt), rng=R)
        cold = (time.perf_counter() - t0) / nd_
        reps = max(3, 600 // nd_)
        t0 = time.perf_counter()
        for _ in range(reps):
            for tt in picks:
                cold_emu.sample(q, draw=int(tt), rng=R)
        warm = (time.perf_counter() - t0) / (reps * nd_)
        print(f"  {nd_:7d}{cold*1e6:15.1f}{warm*1e6:15.1f}"
              f"{hms(warm*PROJECT):>12}")
    del cold_emu

    # ---- what the cache costs ---------------------------------------------
    fresh = fit.sampler()
    t0 = time.perf_counter()
    for t_ in range(fresh.nmcmc):
        fresh.sample(q, draw=t_)
    warm = time.perf_counter() - t0
    print(f"\n  cache warm-up: {fresh.nmcmc} draws, one KD-tree each, "
          f"{warm*1e3:.0f} ms total")
    print(f"  amortised over {PROJECT//1000}k iterations: "
          f"{warm/PROJECT*1e6:.2f} us/eval")

    # ---- a real loop, not a microbenchmark --------------------------------
    print(f"\nend-to-end: {OUTER}-iteration random-walk Metropolis calibration")
    print("-" * 68)
    obs = float(simulator(q, theta_true)[0])
    obs = (obs - y.mean()) / (y.std() if y.std() else 1.0)

    def run_loop(evaluate, iters):
        r = np.random.default_rng(4)
        theta = np.array([0.8, 0.3])
        cur = evaluate(theta, r)
        acc = 0
        t0 = time.perf_counter()
        for _ in range(iters):
            prop = theta + 0.05 * r.standard_normal(2)
            new = evaluate(prop, r)
            # toy Gaussian likelihood against a single field observation
            if np.log(r.uniform()) < -0.5 * ((new - obs) ** 2 - (cur - obs) ** 2) / 0.01:
                theta, cur, acc = prop, new, acc + 1
        return time.perf_counter() - t0, acc / iters

    def eval_sampler(theta, r):
        xq = np.clip(np.atleast_2d(theta), 0, 1)
        return float(emu.sample(xq, draw=int(r.integers(fit.nmcmc)), rng=r)[0, 0])

    def eval_post(theta, r):
        xq = np.clip(np.atleast_2d(theta), 0, 1)
        return float(fit.post_sample(
            xq, draws=int(r.integers(fit.nmcmc)), cores=1, rng=r)[0, 0])

    el_s, acc_s = run_loop(eval_sampler, OUTER)
    short = max(50, OUTER // 20)
    el_p, _ = run_loop(eval_post, short)
    el_p *= OUTER / short  # scale the slow route up rather than wait for it

    print(f"  {'route':<34}{'seconds':>10}{f'x{PROJECT//1000}k':>12}")
    print(f"  {'post_sample(draws=t)':<34}{el_p:9.1f}"
          f"{hms(el_p / OUTER * PROJECT):>12}   (extrapolated from {short})")
    print(f"  {'sampler().sample(draw=t)':<34}{el_s:9.1f}"
          f"{hms(el_s / OUTER * PROJECT):>12}")
    print(f"  acceptance rate {acc_s:.2f}  "
          f"(emulator noise enters every likelihood evaluation)")
    print(f"\n  loop overhead beyond the emulator: "
          f"{(el_s / OUTER - rows[-1][1]) * 1e6:.0f} us/iteration")


if __name__ == "__main__":
    main()
