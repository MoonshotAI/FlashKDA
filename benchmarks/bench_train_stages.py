"""Stage-level timing breakdown: CUDA pipeline vs Triton hosts, fwd and bwd.

Times each pipeline stage with cuda events over N reps after warmup.
Usage: python benchmarks/bench_train_stages.py [B T H D]
"""

import os
import sys

# Prefer a local flash-linear-attention checkout (FLA_REPO) so results are
# measured against the intended Triton reference, not a stale site-packages copy.
_FLA_REPO = os.environ.get("FLA_REPO", "/root/flash-linear-attention")
if os.path.isdir(_FLA_REPO) and _FLA_REPO not in sys.path:
    sys.path.insert(0, _FLA_REPO)

import torch
import torch.nn.functional as F


import fla.ops.kda.chunk_bwd as tri_bwd
import fla.ops.kda.chunk_fwd as tri_fwd
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu as tri_dhu
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h as tri_fwd_h
from fla.ops.gla.chunk import chunk_gla_fwd_o_gk as tri_fwd_o
from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra as tri_bwd_intra
from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra as tri_fwd_intra
from fla.ops.kda.gate import kda_gate_chunk_cumsum as tri_gate_cumsum
from fla.ops.kda.gate import kda_gate_bwd as tri_gate_bwd
from fla.ops.kda.wy_fast import recompute_w_u_fwd as tri_recompute
from fla.ops.utils import chunk_local_cumsum as tri_cumsum

import flash_kda.train as ck

B, T, H, D = (int(x) for x in sys.argv[1:5]) if len(sys.argv) > 4 else (2, 16384, 16, 128)
REPS = 20
device = "cuda"


def bench(fn, reps=REPS):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True)
    e = torch.cuda.Event(True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


torch.manual_seed(42)
dtype = torch.bfloat16
q = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32, device=device), p=2, dim=-1).to(dtype)
k = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32, device=device), p=2, dim=-1).to(dtype)
v = torch.rand(B, T, H, D, dtype=dtype, device=device)
g_raw = torch.randn(B, T, H, D, dtype=dtype, device=device)
beta = torch.randn(B, T, H, dtype=dtype, device=device).sigmoid()
A_log = torch.log(torch.empty(H, dtype=torch.float32, device=device).uniform_(1, 16))
dt_bias = torch.randn(H * D, dtype=torch.float32, device=device)
h0 = torch.randn(B, H, D, D, dtype=torch.float32, device=device)
do = torch.randn(B, T, H, D, dtype=dtype, device=device)
dht = torch.randn(B, H, D, D, dtype=torch.float32, device=device)
scale = D ** -0.5
RCP_LN2 = 1.4426950408889634

# shared intermediates from the Triton fwd (same inputs to both pipelines' later stages)
g = tri_gate_cumsum(g=g_raw, A_log=A_log, dt_bias=dt_bias, scale=RCP_LN2, chunk_size=64, lower_bound=-5.0)
w, u, qg, kg, Aqk, Akk = tri_fwd_intra(q=q, k=k, v=v, gk=g, beta=beta, scale=scale, safe_gate=True)
if qg is None:
    _, _, qg, _ = tri_recompute(k=k, v=v, beta=beta, A=Akk, gk=g, q=q)
h, v_new, ht = tri_fwd_h(k=kg, w=w, u=u, gk=g, initial_state=h0, output_final_state=True)
dAqk, dv = tri_bwd.chunk_kda_bwd_dAv(q=q, k=k, v=v_new, do=do, A=Aqk, scale=scale)
dh, dh0, dv2 = tri_dhu(q=qg, k=kg, w=w, gk=g, h0=h0, dht=dht, do=do, dv=dv, scale=scale)
dq0, dk0, dv3, db0, dg0, dAkk0 = tri_bwd.chunk_kda_bwd_wy_dqkg_fused(
    q=q, k=k, v=v, v_new=v_new, g=g, beta=beta, A=Akk, h=h, do=do, dh=dh, dv=dv2, scale=scale)

rows = []


def add(name, tri_fn, cuda_fn):
    t_tri = bench(tri_fn)
    t_cuda = bench(cuda_fn)
    rows.append((name, t_tri, t_cuda))


add("gate_cumsum",
    lambda: tri_gate_cumsum(g=g_raw, A_log=A_log, dt_bias=dt_bias, scale=RCP_LN2, chunk_size=64, lower_bound=-5.0),
    lambda: ck.kda_gate_chunk_cumsum(g=g_raw, A_log=A_log, dt_bias=dt_bias, scale=RCP_LN2, chunk_size=64, lower_bound=-5.0))
add("fwd_intra",
    lambda: tri_fwd_intra(q=q, k=k, v=v, gk=g, beta=beta, scale=scale, safe_gate=True),
    lambda: ck.chunk_kda_fwd_intra(q=q, k=k, v=v, gk=g, beta=beta, scale=scale, safe_gate=True))
add("recompute_w_u",
    lambda: tri_recompute(k=k, v=v, beta=beta, A=Akk, gk=g, q=q),
    lambda: ck.recompute_w_u_fwd(k=k, v=v, beta=beta, A=Akk, gk=g, q=q))
add("fwd_h",
    lambda: tri_fwd_h(k=kg, w=w, u=u, gk=g, initial_state=h0, output_final_state=True),
    lambda: ck.chunk_gated_delta_rule_fwd_h(k=kg, w=w, u=u, gk=g, initial_state=h0, output_final_state=True))
add("fwd_o",
    lambda: tri_fwd_o(q=q, v=v_new, g=g, A=Aqk, h=h, scale=scale),
    lambda: ck.chunk_gla_fwd_o_gk(q=q, v=v_new, g=g, A=Aqk, h=h, scale=scale))
add("bwd_dAv",
    lambda: tri_bwd.chunk_kda_bwd_dAv(q=q, k=k, v=v_new, do=do, A=Aqk, scale=scale),
    lambda: ck.chunk_kda_bwd_dAv(q=q, k=k, v=v_new, do=do, A=Aqk, scale=scale))
add("bwd_dhu",
    lambda: tri_dhu(q=qg, k=kg, w=w, gk=g, h0=h0, dht=dht, do=do, dv=dv, scale=scale),
    lambda: ck.chunk_gated_delta_rule_bwd_dhu(q=qg, k=kg, w=w, gk=g, h0=h0, dht=dht, do=do, dv=dv, scale=scale))
add("bwd_wy_dqkg",
    lambda: tri_bwd.chunk_kda_bwd_wy_dqkg_fused(q=q, k=k, v=v, v_new=v_new, g=g, beta=beta, A=Akk, h=h, do=do, dh=dh, dv=dv2, scale=scale),
    lambda: ck.chunk_kda_bwd_wy_dqkg_fused(q=q, k=k, v=v, v_new=v_new, g=g, beta=beta, A=Akk, h=h, do=do, dh=dh, dv=dv2, scale=scale))
add("bwd_intra",
    lambda: tri_bwd_intra(q=q, k=k, g=g, beta=beta, dAqk=dAqk, dAkk=dAkk0, dq=dq0, dk=dk0, db=db0, dg=dg0, safe_gate=True),
    lambda: ck.chunk_kda_bwd_intra(q=q, k=k, g=g, beta=beta, dAqk=dAqk, dAkk=dAkk0, dq=dq0, dk=dk0, db=db0, dg=dg0, safe_gate=True))
add("reverse_cumsum",
    lambda: tri_cumsum(dg0, chunk_size=64, reverse=True),
    lambda: ck.chunk_local_cumsum(dg0, chunk_size=64, reverse=True))
add("gate_bwd",
    lambda: tri_gate_bwd(g=g_raw, A_log=A_log, dt_bias=dt_bias, dyg=dg0, lower_bound=-5.0),
    lambda: ck.kda_gate_bwd(g=g_raw, A_log=A_log, dt_bias=dt_bias, dyg=dg0, lower_bound=-5.0))

print(f"\nshape B{B} T{T} H{H} D{D}, {REPS} reps")
print(f"{'stage':16s} {'triton(ms)':>11s} {'cuda(ms)':>9s} {'ratio':>7s}")
tot_t = tot_c = 0.0
for name, t, c in rows:
    print(f"{name:16s} {t:>11.3f} {c:>9.3f} {t/c:>6.2f}x")
    tot_t += t
    tot_c += c
print(f"{'TOTAL':16s} {tot_t:>11.3f} {tot_c:>9.3f} {tot_t/tot_c:>6.2f}x")