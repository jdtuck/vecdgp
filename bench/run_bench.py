"""Race the numba u_entries against a C++/OpenMP transliteration."""
import os, subprocess, sys, time
import numpy as np

sys.path.insert(0, "/home/claude/vecdgp")
from vecdgp.vecchia import create_approx, create_U_values

D = "/home/claude/bench"
THETA, G, V, M = 0.1, 1e-6, 2.5, 25


def timeit(fn, reps):
    best = 1e30
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best


rng = np.random.default_rng(0)
print(f"{'n':>7} {'numba':>10} {'C++ same':>10} {'C++ sym':>10}   {'max |diff|':>11}")
for n in (1000, 5000, 20000, 50000):
    x = rng.random((n, 2))
    ap = create_approx(x, M, rng=rng)
    create_U_values(ap, 1.0, THETA, G, V)  # warm the JIT

    np.array([n, x.shape[1], ap.NN.shape[1], 0], dtype=np.int64).tofile(f"{D}/meta.bin")
    np.array([THETA, G], dtype=np.float64).tofile(f"{D}/par.bin")
    np.ascontiguousarray(ap.x_ord, dtype=np.float64).tofile(f"{D}/x.bin")
    np.ascontiguousarray(ap.NN, dtype=np.int64).tofile(f"{D}/nn.bin")
    np.ascontiguousarray(ap.NN_len, dtype=np.int64).tofile(f"{D}/nl.bin")

    reps = max(3, min(20, int(2e5 / n)))
    t_nb = timeit(lambda: create_U_values(ap, 1.0, THETA, G, V), reps)
    Unb = create_U_values(ap, 1.0, THETA, G, V)

    t_cpp = float(subprocess.run([f"{D}/ubench", D, str(reps), "0"],
                                 capture_output=True, text=True).stdout)
    Ucpp = np.fromfile(f"{D}/u_cpp.bin").reshape(Unb.shape)
    t_sym = float(subprocess.run([f"{D}/ubench", D, str(reps), "1"],
                                 capture_output=True, text=True).stdout)

    print(f"{n:7d} {t_nb*1e3:8.1f}ms {t_cpp*1e3:8.1f}ms {t_sym*1e3:8.1f}ms   "
          f"{np.abs(Unb-Ucpp).max():11.2e}")
