# static weight analysis: magnitude + directional clustering per weight matrix
# every 2D param gets a row view and a column view; 1D params get magnitude stats only
# model-general: the evoke runner supplies the model, this just walks named_parameters
import torch

NUM_PAIRS = 3000
SEED = 42
COS_BINS = 20  # -1..1 in 0.1 divisions
MAG_SIGMA_STEP = 0.5
MAG_SIGMA_RANGE = 4
TAIL_COS = 0.1
SPARK = "▁▂▃▄▅▆▇█"


def sparkline(counts):
    peak = max(counts) or 1
    return "".join(SPARK[min(int(c / peak * 8), 7)] for c in counts)


def magnitude_stats(norms):
    mean = norms.mean().item()
    var = norms.var().item()
    std = var ** 0.5
    lo, hi = mean - MAG_SIGMA_RANGE * std, mean + MAG_SIGMA_RANGE * std
    bins = int(2 * MAG_SIGMA_RANGE / MAG_SIGMA_STEP)
    edges = torch.linspace(lo, hi, bins + 1)[1:-1]
    # bucket 0 = underflow, bucket bins = overflow
    counts = torch.bincount(torch.bucketize(norms, edges), minlength=bins + 1)
    return mean, var, counts.int().tolist()


def cosine_stats(vecs, norms, g):
    n, d = vecs.shape
    u = vecs / norms.clamp_min(1e-12).unsqueeze(-1)
    idx = torch.randint(n, (2, NUM_PAIRS), generator=g)
    mask = idx[0] != idx[1]
    cos = (u[idx[0][mask]] * u[idx[1][mask]]).sum(-1)
    # cos == 1.0 would land in an overflow bucket, nudge it into the last bin
    cos = cos.clamp(-1.0, 1.0 - 1e-7)
    edges = torch.linspace(-1.0, 1.0, COS_BINS + 1)[1:-1]
    hist = torch.bincount(torch.bucketize(cos, edges), minlength=COS_BINS)
    return {
        "hist": hist.int().tolist(),
        "min": cos.min().item(),
        "max": cos.max().item(),
        "frac_pos_tail": (cos > TAIL_COS).float().mean().item(),
        "frac_neg_tail": (cos < -TAIL_COS).float().mean().item(),
        # expected spread for random unit vectors in d dims
        "random_band": 2 / d ** 0.5,
    }


def view_stats(vecs, g):
    norms = vecs.norm(dim=-1)
    mean, var, mag_hist = magnitude_stats(norms)
    centroid_norm = vecs.mean(0).norm().item()
    return {
        "n": vecs.shape[0],
        "dim": vecs.shape[1],
        "mag_mean": mean,
        "mag_var": var,
        "mag_hist": mag_hist,
        # true centroid norm; ratio ~1 = everything points one way, ~0 = spread
        "centroid_norm": centroid_norm,
        "centroid_ratio": centroid_norm / mean,
        "cos": cosine_stats(vecs, norms, g),
    }


def print_view(axis, v):
    c = v["cos"]
    print(f"  {axis}: mag mean {v['mag_mean']:.3f} var {v['mag_var']:.3f} |centroid| {v['centroid_norm']:.2f} ratio {v['centroid_ratio']:.3f}")
    print(f"        mag {sparkline(v['mag_hist'])}")
    print(f"        cos {sparkline(c['hist'])}  min {c['min']:.3f} max {c['max']:.3f}  >{TAIL_COS}: {c['frac_pos_tail']:.1%}  <-{TAIL_COS}: {c['frac_neg_tail']:.1%}  random ±{c['random_band']:.3f}")


def analyze_model(model):
    report = {}
    for name, p in model.named_parameters():
        # stats on cpu: fp32 copies of big mats don't fit next to the model on 8gb
        w = p.detach().float().cpu()
        print(f"{name} {tuple(w.shape)}")
        if w.ndim == 2:
            g = torch.Generator().manual_seed(SEED)
            entry = {"rows": view_stats(w, g), "cols": view_stats(w.T, g)}
            print_view("rows", entry["rows"])
            print_view("cols", entry["cols"])
        else:
            mean, var, hist = magnitude_stats(w.flatten().abs())
            entry = {"mag_mean": mean, "mag_var": var, "mag_hist": hist}
            print(f"  1D: mag mean {mean:.3f} var {var:.3f}")
            print(f"      mag {sparkline(hist)}")
        report[name] = entry
    return report
