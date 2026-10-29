"""Build the dashboard data file from classified + raw reviews.

Run:  python dashboard/build_dashboard_data.py --src data/classified_sample.csv
Out:  dashboard/data/my_dashboard_data.js   (window.DASHBOARD_DATA = {...})
     The shipped dashboard_data.js was built the same way from all German reviews.

Re-run whenever new reviews are scraped and classified; the page needs no change.
Rules (also shown in the page footer):
  * Sources: Google Play (German-language, DACH) and App Store (DE+AT+CH storefronts).
  * Focal app: komoot. Benchmark: equal-weighted mean of competitor app means
    (apps with >= MIN_APP_N reviews that month), so one big app can't dominate.
  * A monthly point is shown only if n >= MIN_N; otherwise it is a gap.
  * Last month is flagged partial if data ends > 3 days before month end.
  * Themes: binary classifier labels (18 themes). Rating gap = mean stars with
    the theme minus without it; association, not effect.
  * Topics: German tokens/bigrams; trending = log-odds with informative Dirichlet
    prior (Monroe et al. 2008), recent 3 complete months vs the 12 months before.
"""
import argparse, json, math, re, collections, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="data/classified_sample.csv")   # one row per review: app, store, country, date, rating, title, text, theme 0/1 + p_theme
ap.add_argument("--out", default="dashboard/data/my_dashboard_data.js")
args, _ = ap.parse_known_args()
OUT = Path(args.out)
FOCAL, START = "komoot", "2022-01"
MIN_N, MIN_APP_N = 20, 10
RECENT_M, BASE_M = 3, 12

STOP = set("""aber alle allem allen aller alles als also am an ander andere anderen anderer anderes auch auf aus bei beim bin bis bist bitte da dabei dadurch daher damit dann darauf darf das dass dazu dem den denen denn der des deshalb dich die dies diese diesem diesen dieser dieses dir doch dort du durch ein eine einem einen einer eines einfach einmal er es etc etwas euch euer für gab gar gibt gut habe haben hat hatte hätte hier hin ich ihm ihn ihr ihre im immer in ist ja jetzt kann kein keine keinen konnte könnte man mehr mein meine meinem meinen meiner mich mir mit muss nach nicht nichts noch nun nur ob oder ohne schon sehr sein seine seit sich sie sind so soll sollte sonst um und uns unser unter viel vom von vor war waren was weil wenn wer werden wie wieder wir wird wo wurde wurden zu zum zur über leider mal ganz ab total richtig echt geht gehen ging einen beim habe wäre sowie bzw dass daß komoot app apps eigentlich wirklich immer solche merkt""".split())
DE_FN = set("und ist die der das nicht ich mit sehr auch aber für ein eine zu es sich auf".split())
FOREIGN_FN = set("the and is not this it very but for with les est une pas très pour avec mais che non per con molto".split())
TOKEN = re.compile(r"[a-zäöüß]{3,}")

def months_between(a, b):
    p = pd.period_range(a, b, freq="M"); return [str(x) for x in p]

# ---------------------------------------------------------------- load
d = pd.read_csv(args.src)
THEMES = [c for c in d.columns if f"p_{c}" in d.columns]
d["date"] = d.date.astype(str).str[:10]
d = d[d.date >= START].copy()
d["m"] = d.date.str[:7]
d["doc"] = (d.title.fillna("") + " " + d.text.fillna("")).str.lower()

last_date = d.date.max()
last_m = last_date[:7]
end_of_month = (pd.Period(last_m, "M").end_time.date())
partial = (end_of_month - dt.date.fromisoformat(last_date)).days > 3
MONTHS = months_between(START, last_m)
complete = MONTHS[:-1] if partial else MONTHS
cur_m = complete[-1]
recent = complete[-RECENT_M:]
base = complete[-(RECENT_M + BASE_M):-RECENT_M]
year_ago = [str(pd.Period(m, "M") - 12) for m in recent]

STORES = {"all": d, "play": d[d.store == "play"], "apple": d[d.store == "apple"]}
APPS = sorted(d.app.unique())

def mstats(x):
    g = x.groupby("m").rating.agg(["mean", "count"])
    g["s1"] = x.assign(o=x.rating.eq(1)).groupby("m").o.mean()
    g["s5"] = x.assign(o=x.rating.eq(5)).groupby("m").o.mean()
    return g.reindex(MONTHS)

def r(v, k=3):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), k)

# ---------------------------------------------------------------- ratings
ratings = {}
for s, x in STORES.items():
    per_app = {a: mstats(x[x.app == a]) for a in APPS}
    # benchmark: equal-weight mean of competitor app means (n>=MIN_APP_N)
    comp = [a for a in APPS if a != FOCAL]
    M = pd.DataFrame({a: per_app[a]["mean"].where(per_app[a]["count"] >= MIN_APP_N) for a in comp})
    bench = M.mean(axis=1)
    bench_apps = M.notna().sum(axis=1)
    bench_n = sum(per_app[a]["count"].fillna(0) for a in comp)
    f = per_app[FOCAL]
    ratings[s] = {
        "focal": {"mean": [r(v) if c >= MIN_N else None for v, c in zip(f["mean"], f["count"].fillna(0))],
                  "n": [int(c) if c == c else 0 for c in f["count"]],
                  "share1": [r(v) if c >= MIN_N else None for v, c in zip(f["s1"], f["count"].fillna(0))],
                  "share5": [r(v) if c >= MIN_N else None for v, c in zip(f["s5"], f["count"].fillna(0))]},
        "bench": {"mean": [r(v) if k >= 3 else None for v, k in zip(bench, bench_apps)],
                  "apps": [int(k) for k in bench_apps], "n": [int(v) for v in bench_n]},
        "apps": {a: {"mean": [r(v) if c >= MIN_APP_N else None for v, c in zip(per_app[a]["mean"], per_app[a]["count"].fillna(0))],
                     "n": [int(c) if c == c else 0 for c in per_app[a]["count"]]} for a in APPS},
    }
    # app league table: recent window vs year ago
    rows = []
    for a in APPS:
        xa = x[x.app == a]
        rn, yn = xa[xa.m.isin(recent)], xa[xa.m.isin(year_ago)]
        if len(rn) < MIN_APP_N: continue
        rows.append({"app": a, "mean": r(rn.rating.mean(), 2), "n": len(rn),
                     "yoy": r(rn.rating.mean() - yn.rating.mean(), 2) if len(yn) >= MIN_APP_N else None,
                     "share1": r(rn.rating.eq(1).mean(), 3)})
    ratings[s]["league"] = sorted(rows, key=lambda z: -z["mean"])

# ---------------------------------------------------------------- themes (focal)
themes = {}
for s, x in STORES.items():
    k = x[x.app == FOCAL]
    c = x[x.app != FOCAL]
    monthly = k.groupby("m")[THEMES].mean().reindex(MONTHS)
    cnt = k.groupby("m").size().reindex(MONTHS).fillna(0)
    win = k[k.m.isin(complete[-12:])]
    rows = []
    for t in THEMES:
        w, wo = win[win[t] == 1], win[win[t] == 0]
        sh = lambda z: z[t].mean() if len(z) else np.nan
        kr, ky = k[k.m.isin(recent)], k[k.m.isin(year_ago)]
        kp = k[k.m.isin(complete[-2 * RECENT_M:-RECENT_M])]
        cr = c[c.m.isin(recent)]
        gap = w.rating.mean() - wo.rating.mean() if len(w) >= 10 else np.nan
        rows.append({"theme": t, "share12": r(sh(win), 4), "n12": int(len(w)),
                     "gap": r(gap, 2), "mean_with": r(w.rating.mean(), 2) if len(w) >= 10 else None,
                     "impact": r(sh(win) * gap, 4) if gap == gap else None,
                     "share_recent": r(sh(kr), 4), "share_prev": r(sh(kp), 4), "share_yago": r(sh(ky), 4),
                     "bench_recent": r(sh(cr), 4),
                     "series": [r(v, 4) if n >= MIN_N else None for v, n in zip(monthly[t], cnt)]})
    # theme co-occurrence lift in the 12-month window
    X = win[THEMES].values.astype(float)
    nwin = len(win); co = X.T @ X; p = X.mean(0)
    lift = [[r((co[i, j] / nwin) / (p[i] * p[j]), 2) if p[i] * p[j] > 0 and i != j and co[i, j] >= 5 else None
             for j in range(len(THEMES))] for i in range(len(THEMES))]
    themes[s] = {"rows": rows, "cooc": {"lift": lift, "count": co.astype(int).tolist()}, "n12": nwin,
                 "mean12": r(win.rating.mean(), 2)}

# ---------------------------------------------------------------- topics (focal)
def tokens(doc):
    t = TOKEN.findall(doc)
    uni = {w for w in t if w not in STOP}
    bi = {f"{a} {b}" for a, b in zip(t, t[1:]) if a not in STOP and b not in STOP and a != b}
    return uni | bi

def logodds(ca, na, cb, nb, prior, a0):
    # Monroe et al. 2008, informative Dirichlet prior; doc-frequency counts
    out = {}
    for w in set(ca) | set(cb):
        aw = prior.get(w, 0) * a0 + 0.01
        ya, yb = ca.get(w, 0), cb.get(w, 0)
        d_ = math.log((ya + aw) / (na + a0 - ya - aw)) - math.log((yb + aw) / (nb + a0 - yb - aw))
        out[w] = d_ / math.sqrt(1 / (ya + aw) + 1 / (yb + aw))
    return out

topics = {}
for s, x in STORES.items():
    k = x[x.app == FOCAL].copy()
    w_ = k.doc.str.findall(r"[a-zäöüßéèàç]+")
    k = k[[sum(t in DE_FN for t in ws) >= sum(t in FOREIGN_FN for t in ws) for ws in w_]]  # drop non-German text
    k["tok"] = k.doc.map(tokens)
    R, B = k[k.m.isin(recent)], k[k.m.isin(base)]
    if len(R) < 150 or len(B) < 300:
        topics[s] = {"insufficient": True, "n_recent": len(R), "n_base": len(B)}; continue
    cr = collections.Counter(w for ts in R.tok for w in ts)
    cb = collections.Counter(w for ts in B.tok for w in ts)
    call = collections.Counter(w for ts in k.tok for w in ts)
    tot = sum(call.values()); prior = {w: c / tot for w, c in call.items()}
    a0 = 1000.0
    z = logodds(cr, len(R), cb, len(B), prior, a0)
    minc = max(5, int(0.004 * len(R)))
    cand = [w for w in z if cr.get(w, 0) >= minc or cb.get(w, 0) * len(R) / len(B) >= minc]
    spark_m = complete[-24:]
    km = k[k.m.isin(spark_m)]
    by_m = {m: g.tok.tolist() for m, g in km.groupby("m")}
    def term_row(w):
        ws = k.tok.map(lambda ts: w in ts)
        sub = k[ws & k.m.isin(complete[-12:])]
        # theme most associated with the term (lift among docs with term)
        tl = {}
        allw = k[k.m.isin(complete[-12:])]
        for t in THEMES:
            pt = allw[t].mean()
            if pt > 0 and sub[t].sum() >= 8: tl[t] = sub[t].mean() / pt
        top_t = max(tl, key=tl.get) if tl else None
        return {"term": w, "z": r(z[w], 2),
                "share_recent": r(cr.get(w, 0) / len(R), 4), "share_base": r(cb.get(w, 0) / len(B), 4),
                "n_recent": cr.get(w, 0), "mean_star": r(sub.rating.mean(), 2) if len(sub) >= 5 else None,
                "theme": top_t, "theme_lift": r(tl.get(top_t), 2) if top_t else None,
                "spark": [r(sum(w in ts for ts in by_m.get(m, [])) / max(1, len(by_m.get(m, []))), 4) for m in spark_m]}
    rising = sorted([w for w in cand if z[w] > 0], key=lambda w: -z[w])[:30]
    falling = sorted([w for w in cand if z[w] < 0], key=z.get)[:15]
    top = [w for w, _ in cr.most_common(400) if " " not in w][:60]
    # keyword pair co-occurrence: recent vs base, among recent top unigrams + rising unigrams
    vocab = list(dict.fromkeys([w for w in rising if " " not in w] + top))[:70]
    def pairs(df):
        c = collections.Counter()
        for ts in df.tok:
            v = sorted(w for w in vocab if w in ts)
            for i in range(len(v)):
                for j in range(i + 1, len(v)): c[(v[i], v[j])] += 1
        return c
    pr, pb = pairs(R), pairs(B)
    ur = {w: cr.get(w, 0) for w in vocab}
    prow = []
    for (a, b), c in pr.items():
        if c < max(5, minc): continue
        lift = (c / len(R)) / ((ur[a] / len(R)) * (ur[b] / len(R)))
        sb = pb.get((a, b), 0) / len(B); sr = c / len(R)
        prow.append({"a": a, "b": b, "n": c, "share_recent": r(sr, 4), "share_base": r(sb, 4),
                     "growth": r(sr / sb, 2) if sb > 0 else None, "lift": r(lift, 2)})
    prow = sorted(prow, key=lambda p: -(p["share_recent"] - (p["share_base"] or 0)))[:25]
    topics[s] = {"n_recent": len(R), "n_base": len(B), "spark_months": spark_m,
                 "rising": [term_row(w) for w in rising], "falling": [term_row(w) for w in falling],
                 "pairs": prow}

out = {
    "meta": {"focal": FOCAL, "built": dt.datetime.now().isoformat(timespec="minutes"),
             "data_through": last_date, "partial_last_month": partial, "current_month": cur_m,
             "months": MONTHS, "recent": recent, "base": base, "year_ago": year_ago,
             "min_n": MIN_N, "min_app_n": MIN_APP_N, "themes": THEMES, "apps": APPS,
             "n_reviews": {s: int(len(x)) for s, x in STORES.items()},
             "n_focal": {s: int((x.app == FOCAL).sum()) for s, x in STORES.items()},
             "sources": {"play": "Google Play · German-language reviews (DACH)",
                         "apple": "App Store · DE + AT + CH storefronts"}},
    "ratings": ratings, "themes": themes, "topics": topics,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("window.DASHBOARD_DATA = " + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size/1e3:.0f} kB) · through {last_date} · current {cur_m} · partial={partial}")
