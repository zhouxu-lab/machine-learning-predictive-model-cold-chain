from pathlib import Path
import sys
_PACKAGE = next(p for p in Path(__file__).resolve().parents if (p/'05_Data').is_dir())
sys.path.insert(0,str(_PACKAGE/'04_Code'))
sys.path.insert(0,str(_PACKAGE/'04_Code/Shared'))
from package_paths import PACKAGE_ROOT, CODE, DATA, RESULTS
"""
M1 / M2: deep models for growth prediction under dynamic temperature profiles.

  gru        : GRU over the resampled temperature profile + covariates (pure DL, M1)
  node       : physics-informed neural ODE with Baranyi structure, NO history term (ablation)
  node_hist  : same, plus a learned temperature-history term acting on the physiological
               state q (lag re-induction after temperature shifts)  -> main model (M2)

State (per record) on a fixed grid of G steps between first observation t0 and last obs:
  y  : ln(N) - ln(N0)          (growth in natural-log units)
  lq : ln(q)                   (Baranyi physiological state)
  d y /dt = mu(T,x) * q/(1+q) * (1 - exp(y - ymax(x))) - kd(T,x)
  d lq/dt = mu(T,x) + g(T, dT/dt, x)      [g == 0 for 'node']
mu, kd bounded in (0, 3) h-1 by scaled sigmoids; ymax(x) = ln(10) * (softplus(NN)+ 1) so ymax in log10 >= 1 above y0.
Trained on static + dynamic train records, evaluated on dynamic test records
with leave-sources-out folds shared with the GBM baseline.

usage: python deep_models.py <gru|transformer|node|node_hist> [epochs] [tag] [lr]   (env SEED, TORCH_THREADS)
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from models_common import load, build_dynamic_targets, metrics, temperature_function, source_folds, HERE
from baseline_gbm import COND_FLAGS
RES = str(RESULTS/'Training/Preparation')
os.makedirs(RES, exist_ok=True)
G = 96              # grid points per record
LN10 = float(np.log(10))
SEED = int(os.environ.get("SEED", 0))
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", max(1, os.cpu_count() - 2))))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def prepare(rec, cur, pro):
    tgt = build_dynamic_targets(rec, cur, pro, kinds=("static", "dynamic"))
    ct = tgt.conditions_text.fillna("").str.lower()
    tgt["nacl_pct"] = ct.str.extract(r"sodium chloride[^:]*:\s*([\d.]+)")[0].astype(float).fillna(0).clip(0, 25) / 10
    for k, pat in COND_FLAGS.items():
        tgt["c_" + k] = ct.str.contains(pat, regex=True).astype(float)
    orgs = {o: i for i, o in enumerate(sorted(tgt.organism.fillna("unk").unique()))}
    mats = {m: i for i, m in enumerate(sorted(tgt.food_category.fillna("unk").unique()))}
    Tpro = pro[pro.variable == "Temperatures"]
    Tpro_g = {k: v for k, v in Tpro.groupby("record_id")}
    recs = []
    for rid, g in tgt.groupby("record_id", sort=False):
        r0 = g.iloc[0]
        t0, tend = float(r0.t0), float(g.time_h.max())
        Tc = r0.temperature if pd.notna(r0.temperature) else r0.T_mean
        Tfun = temperature_function(Tpro_g.get(rid) if r0.kind == "dynamic" else None, Tc)
        grid = np.linspace(t0, tend, G)
        T = Tfun(grid).astype(np.float32)
        x = np.array([(r0.ph if pd.notna(r0.ph) else 6.5) - 6.5, float(pd.isna(r0.ph)),
                      ((r0.aw if pd.notna(r0.aw) else 0.99) - 0.97) * 30, float(pd.isna(r0.aw)),
                      r0.nacl_pct, float(r0.matrix_is_broth)] + [r0["c_" + k] for k in COND_FLAGS], dtype=np.float32)
        obs_t = (g.time_h.values - t0) / max(tend - t0, 1e-6) * (G - 1)      # fractional grid index
        recs.append(dict(record_id=rid, kind=r0.kind, org=orgs[r0.organism if pd.notna(r0.organism) else "unk"],
                         mat=mats[r0.food_category if pd.notna(r0.food_category) else "unk"], x=x, T=T,
                         dt=np.float32((tend - t0) / (G - 1)), y0=np.float32(r0.logc0_obs),
                         obs_idx=obs_t.astype(np.float32), obs_y=(g.logc.values - r0.logc0_obs).astype(np.float32),
                         obs_time=g.time_h.values, source_key=r0.source_key if pd.notna(r0.source_key) else "unk",
                         coldchain=bool(r0.coldchain), organism=r0.organism))
    return recs, orgs, mats
class Cov(nn.Module):
    def __init__(self, n_org, n_mat, n_x, d=32):
        super().__init__()
        self.eo = nn.Embedding(n_org, 16); self.em = nn.Embedding(n_mat, 8)
        self.fx = nn.Sequential(nn.Linear(n_x + 24, d), nn.SiLU(), nn.Linear(d, d))
    def forward(self, x, org, mat):
        return self.fx(torch.cat([x, self.eo(org), self.em(mat)], -1))
class NeuralBaranyi(nn.Module):
    """Physics-informed neural Baranyi ODE. Rate networks are evaluated once on the whole
    temperature grid (they depend only on covariates, T and dT/dt); the RK4 loop is then
    purely element-wise, which keeps GPU kernel launches low."""
    def __init__(self, n_org, n_mat, n_x, history=True, d=32):
        super().__init__()
        self.cov = Cov(n_org, n_mat, n_x, d)
        self.rate = nn.Sequential(nn.Linear(d + 3, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))   # mu, kd
        self.ymax = nn.Sequential(nn.Linear(d + 1, 32), nn.SiLU(), nn.Linear(32, 1))
        self.q0 = nn.Sequential(nn.Linear(d, 32), nn.SiLU(), nn.Linear(32, 1))
        self.history = history
        if history:
            # history term g = a + b*tanh(ln q / 5) + e*tanh(y / 5), with (a, b, e) functions of (x, T, dT/dt)
            self.hist = nn.Sequential(nn.Linear(d + 2, 32), nn.SiLU(), nn.Linear(32, 3))

    def rates(self, c, T, dT):
        """c: (n,d); T, dT: (n,) or (n,G). Returns mu, kd in ln/h with the shape of T."""
        Tn = (T - 20) / 20
        if T.dim() == 2:
            c = c[:, None, :].expand(-1, T.shape[1], -1)
        out = self.rate(torch.cat([c, Tn[..., None], Tn[..., None] ** 2, dT[..., None]], -1))
        mu = 3.0 * torch.sigmoid(out[..., 0] - 4.0)
        kd = 3.0 * torch.sigmoid(out[..., 1] - 6.0)
        return mu, kd

    max_sub, sub_h = 8, 4.0   # Per-record RK4 rule; overridden for numerical sensitivity checks.

    def forward(self, T, x, org, mat, dt, y0):
        if not torch.isfinite(dt).all() or (dt <= 0).any():
            raise ValueError('ODE time steps must be finite and positive')
        c = self.cov(x, org, mat)
        n = T.shape[0]
        ymax = LN10 * (nn.functional.softplus(self.ymax(torch.cat([c, y0[:, None]], -1))[:, 0]) + 1.0)
        lq = self.q0(c)[:, 0]
        y = torch.zeros(n, device=T.device)
        dTdt = (torch.gradient(T, dim=1)[0] / dt[:, None] * 10).clamp(-50, 50)   # C per 10 h, clipped
        mu_g, kd_g = self.rates(c, T, dTdt)                                       # (n,G)
        if self.history:
            cg = c[:, None, :].expand(-1, G, -1)
            hg = self.hist(torch.cat([cg, ((T - 20) / 20)[..., None], dTdt[..., None]], -1))   # (n,G,3)
            a_g, b_g, e_g = hg[..., 0], hg[..., 1], hg[..., 2]
        # A record's integration rule must not depend on other batch members.
        n_sub = torch.ceil(dt / self.sub_h).clamp(1, self.max_sub)
        max_steps = int(n_sub.max().item())
        h = dt / n_sub
        Y = [y]
        for k in range(G - 1):
            def f(y, lq, mu, kd, a, b, e):
                q = torch.exp(lq.clamp(-20, 20))
                alpha = q / (1 + q)
                dy = mu * alpha * (1 - torch.exp((y - ymax).clamp(max=0))) - kd
                dlq = mu
                if self.history:
                    dlq = dlq + 0.1 * (a + b * torch.tanh(lq / 5) + e * torch.tanh(y / 5))
                return dy, dlq
            def at(w):   # linear interpolation of the grid fields between k and k+1
                mu = mu_g[:, k] + w * (mu_g[:, k + 1] - mu_g[:, k]); kd = kd_g[:, k] + w * (kd_g[:, k + 1] - kd_g[:, k])
                if self.history:
                    a = a_g[:, k] + w * (a_g[:, k + 1] - a_g[:, k]); b = b_g[:, k] + w * (b_g[:, k + 1] - b_g[:, k]); e = e_g[:, k] + w * (e_g[:, k + 1] - e_g[:, k])
                else:
                    a = b = e = None
                return mu, kd, a, b, e
            for j in range(max_steps):
                active = j < n_sub
                r0 = at((j / n_sub).clamp(max=1)); rm = at(((j + 0.5) / n_sub).clamp(max=1)); r1 = at(((j + 1) / n_sub).clamp(max=1))
                k1y, k1q = f(y, lq, *r0)
                k2y, k2q = f(y + 0.5 * h * k1y, lq + 0.5 * h * k1q, *rm)
                k3y, k3q = f(y + 0.5 * h * k2y, lq + 0.5 * h * k2q, *rm)
                k4y, k4q = f(y + h * k3y, lq + h * k3q, *r1)
                y = torch.where(active, (y + h / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)).clamp(-35, 35), y)
                lq = torch.where(active, (lq + h / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)).clamp(-30, 30), lq)
            Y.append(y)
        return torch.stack(Y, 1) / LN10          # log10 units, relative to y0
