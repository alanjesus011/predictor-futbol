"""
Predictor de fútbol — corre solo en GitHub cada día.

Para cada competencia: descarga resultados (y momios cuando existen), entrena un
modelo Dixon-Coles con prueba honesta (backtest), calcula el peso óptimo de los
momios y escribe docs/data/<id>.json para la app. Champions usa el Elo de ClubElo,
que pone a todos los clubes de Europa en la misma escala.
"""
import io, json, os, sys, time, traceback, urllib.request
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

BASE = os.path.dirname(os.path.abspath(__file__))
SALIDA = os.path.join(BASE, "docs", "data")
HOY = pd.Timestamp.now(tz="America/Mexico_City").tz_localize(None).normalize()
GRID = 10

FD = "https://www.football-data.co.uk"
LIGAS = [
    dict(id="liga_mx",      nombre="Liga MX",       bandera="🇲🇽", tipo="dc", fuente="new",  codigo="MEX", liga="Liga MX"),
    dict(id="selecciones",  nombre="Selecciones",   bandera="🌎", tipo="dc", fuente="intl"),
    dict(id="espana",       nombre="LaLiga",        bandera="🇪🇸", tipo="dc", fuente="main", codigo="SP1"),
    dict(id="inglaterra",   nombre="Premier League", bandera="🏴", tipo="dc", fuente="main", codigo="E0"),
    dict(id="italia",       nombre="Serie A",       bandera="🇮🇹", tipo="dc", fuente="main", codigo="I1"),
    dict(id="portugal",     nombre="Primeira Liga", bandera="🇵🇹", tipo="dc", fuente="main", codigo="P1"),
    dict(id="paises_bajos", nombre="Eredivisie",    bandera="🇳🇱", tipo="dc", fuente="main", codigo="N1"),
    dict(id="champions",    nombre="Champions",     bandera="⭐", tipo="elo"),
]

# Ajustes por tipo de competencia (la prueba elige la mejor combinación)
AJUSTES = {
    "clubes":      dict(anios=4, xi=[0.0015, 0.003, 0.006], reg=[1, 5, 15], k_elo=[0.0, 0.15], periodo="W", dias_activo=150),
    "selecciones": dict(anios=8, xi=[0.0007, 0.0015],       reg=[2, 8],     k_elo=[0.0, 0.15, 0.3], periodo="M", dias_activo=1000),
}
DIAS_PRUEBA = 365


# ============================== DESCARGAS ==============================
def descargar(url):
    """Descarga texto. Con la variable PRUEBA_DIR se leen archivos locales (para pruebas)."""
    d = os.environ.get("PRUEBA_DIR")
    if d:
        ruta = os.path.join(d, url.split("://", 1)[1].replace("/", "_"))
        if not os.path.exists(ruta):
            raise FileNotFoundError(ruta)
        return open(ruta, encoding="latin-1").read()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (predictor-futbol)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
    for enc in ("utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass


def leer_csv(url):
    return pd.read_csv(io.StringIO(descargar(url)), on_bad_lines="skip")


def col(df, *opciones):
    return next((c for c in opciones if c in df.columns), None)


def momios_a_prob(df):
    """Probabilidades sin comisión a partir de momios: cierre promedio > Pinnacle > promedio > Bet365."""
    cols = [col(df, f"AvgC{x}", f"PSC{x}", f"Avg{x}", f"PS{x}", f"B365{x}") for x in "HDA"]
    if not all(cols):
        return np.full((len(df), 3), np.nan), None
    inv = 1 / df[cols].apply(pd.to_numeric, errors="coerce").values
    return inv / inv.sum(1, keepdims=True), cols


def momios_decimales(df):
    cols = [col(df, f"Avg{x}", f"PS{x}", f"B365{x}", f"AvgC{x}", f"PSC{x}", f"Max{x}") for x in "HDA"]
    if not all(cols):
        return [None] * len(df)
    v = df[cols].apply(pd.to_numeric, errors="coerce").values
    return [list(map(float, r)) if np.all(r > 1) else None for r in v]


def fechas(serie):
    """Lee fechas tanto en formato 2024-08-15 como 15/08/2024 o 15/08/24."""
    s = serie.astype(str).str.strip()
    iso = s.str.match(r"^\d{4}-")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if iso.any():
        out[iso] = pd.to_datetime(s[iso].str[:10], format="%Y-%m-%d", errors="coerce")
    if (~iso).any():
        out[~iso] = pd.to_datetime(s[~iso], dayfirst=True, errors="coerce", format="mixed")
    return out


def normalizar(raw, h, a, gh, ga, neutral=None, peso=None, torneo=None):
    df = pd.DataFrame({
        "fecha": fechas(raw["Date"] if "Date" in raw.columns else raw["date"]).values,
        "local": raw[h].astype(str).str.strip(), "visita": raw[a].astype(str).str.strip(),
        "gl": pd.to_numeric(raw[gh], errors="coerce"), "gv": pd.to_numeric(raw[ga], errors="coerce"),
    })
    df["neutral"] = False if neutral is None else neutral.values
    df["peso"] = 1.0 if peso is None else peso.values
    df["torneo"] = "" if torneo is None else torneo.values
    return df


def cargar_liga_new(cfg):
    raw = leer_csv(f"{FD}/new/{cfg['codigo']}.csv")
    if "League" in raw.columns:
        raw = raw[raw["League"].astype(str).str.strip() == cfg["liga"]]
    df = normalizar(raw, col(raw, "Home", "HomeTeam"), col(raw, "Away", "AwayTeam"), col(raw, "HG", "FTHG"), col(raw, "AG", "FTAG"))
    df[["casa_1", "casa_x", "casa_2"]], _ = momios_a_prob(raw)
    return df


def temporadas(anios):
    y = HOY.year if HOY.month >= 7 else HOY.year - 1
    return [f"{a % 100:02d}{(a + 1) % 100:02d}" for a in range(y - anios, y + 1)]


def cargar_liga_main(cfg, anios):
    partes = []
    for t in temporadas(anios):
        try:
            raw = leer_csv(f"{FD}/mmz4281/{t}/{cfg['codigo']}.csv")
        except Exception:
            continue
        raw = raw.dropna(subset=[col(raw, "HomeTeam", "Home")])
        df = normalizar(raw, col(raw, "HomeTeam", "Home"), col(raw, "AwayTeam", "Away"), col(raw, "FTHG", "HG"), col(raw, "FTAG", "AG"))
        df[["casa_1", "casa_x", "casa_2"]], _ = momios_a_prob(raw)
        partes.append(df)
    if not partes:
        raise RuntimeError("No se pudo descargar ninguna temporada")
    return pd.concat(partes, ignore_index=True)


INTL_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def cargar_intl():
    raw = leer_csv(INTL_URL)
    torneo = raw["tournament"].astype(str)
    neutral = raw["neutral"].astype(str).str.upper().isin(["TRUE", "1"])
    peso = np.where(torneo == "Friendly", 0.5, 1.0)
    df = normalizar(raw, "home_team", "away_team", "home_score", "away_score", neutral, pd.Series(peso), torneo)
    df[["casa_1", "casa_x", "casa_2"]] = np.nan
    return df


def proximos(cfg, modelo_equipos):
    """Próximos partidos (con momios cuando el sitio los publica)."""
    try:
        if cfg["fuente"] == "main":
            raw = leer_csv(f"{FD}/fixtures.csv")
            raw = raw[raw[col(raw, "Div")].astype(str).str.strip() == cfg["codigo"]]
            h, a = col(raw, "HomeTeam", "Home"), col(raw, "AwayTeam", "Away")
            neutral = [False] * len(raw)
        elif cfg["fuente"] == "new":
            raw = leer_csv(f"{FD}/new_league_fixtures.csv")
            if "League" in raw.columns:
                raw = raw[raw["League"].astype(str).str.strip() == cfg["liga"]]
            h, a = col(raw, "Home", "HomeTeam"), col(raw, "Away", "AwayTeam")
            neutral = [False] * len(raw)
        elif cfg["fuente"] == "intl":
            raw = leer_csv(INTL_URL)
            raw = raw[raw["home_score"].isna()]
            raw = raw.rename(columns={"date": "Date"})
            h, a = "home_team", "away_team"
            neutral = raw["neutral"].astype(str).str.upper().isin(["TRUE", "1"]).tolist()
        else:
            return []
        fch = fechas(raw["Date"])
        mom = momios_decimales(raw)
        out = []
        for f, l, v, n, m in zip(fch, raw[h].astype(str).str.strip(), raw[a].astype(str).str.strip(), neutral, mom):
            if pd.isna(f) or f < HOY - pd.Timedelta(days=1) or f > HOY + pd.Timedelta(days=45):
                continue
            if l in modelo_equipos and v in modelo_equipos:
                out.append(dict(fecha=f.strftime("%Y-%m-%d"), local=l, visita=v, neutral=bool(n), momios=m))
        return sorted(out, key=lambda x: x["fecha"])[:40]
    except Exception as e:
        print(f"   (sin próximos partidos: {e})")
        return []


# ============================== ELO ==============================
def elo_clubes(df, k=20, ventaja=60, inicial=1500):
    r, el_, ev_ = {}, [], []
    for loc, vis, gl, gv in zip(df.local, df.visita, df.gl, df.gv):
        el, ev = r.get(loc, inicial), r.get(vis, inicial)
        el_.append(el); ev_.append(ev)
        esp = 1 / (1 + 10 ** (-(el + ventaja - ev) / 400))
        real = 1.0 if gl > gv else (0.5 if gl == gv else 0.0)
        c = k * (np.log(abs(gl - gv) + 1) + 1) * (real - esp)
        r[loc], r[vis] = el + c, ev - c
    return np.array(el_), np.array(ev_), r


def k_torneo(t):
    t = str(t)
    if t == "FIFA World Cup": return 60
    if t in ("UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup", "Gold Cup", "CONCACAF Championship"): return 50
    if "qualification" in t or "Nations League" in t: return 40
    if t == "Friendly": return 20
    return 30


def elo_selecciones(df, inicial=1500, ventaja=100):
    """Elo al estilo eloratings.net: peso por torneo, margen de goles y sede neutral."""
    r, el_, ev_ = {}, [], []
    for loc, vis, gl, gv, neu, t in zip(df.local, df.visita, df.gl, df.gv, df.neutral, df.torneo):
        el, ev = r.get(loc, inicial), r.get(vis, inicial)
        el_.append(el); ev_.append(ev)
        esp = 1 / (1 + 10 ** (-(el + (0 if neu else ventaja) - ev) / 400))
        real = 1.0 if gl > gv else (0.5 if gl == gv else 0.0)
        d = abs(gl - gv); g = 1 if d <= 1 else (1.5 if d == 2 else (11 + d) / 8)
        c = k_torneo(t) * g * (real - esp)
        r[loc], r[vis] = el + c, ev - c
    return np.array(el_), np.array(ev_), r


# ============================== MODELO DIXON-COLES ==============================
def ajustar(df, fecha_corte, xi, reg, anios, min_partidos=1):
    """Dixon-Coles: ataque, defensa, ventaja de local (solo si no es neutral), goles base y rho. Gradiente exacto."""
    t = df[(df.fecha < fecha_corte) & (df.fecha >= fecha_corte - pd.Timedelta(days=int(365 * anios)))]
    if min_partidos > 1:
        cuenta = pd.concat([t.local, t.visita]).value_counts()
        ok = set(cuenta[cuenta >= min_partidos].index)
        t = t[t.local.isin(ok) & t.visita.isin(ok)]
    eq = sorted(set(t.local) | set(t.visita)); n = len(eq); idx = {e: i for i, e in enumerate(eq)}
    hi, ai = t.local.map(idx).values, t.visita.map(idx).values
    gl, gv = t.gl.values.astype(float), t.gv.values.astype(float)
    hf = (~t.neutral.values.astype(bool)).astype(float)
    w = np.exp(-xi * (fecha_corte - t.fecha).dt.days.values) * t.peso.values
    m00, m01, m10, m11 = (gl == 0) & (gv == 0), (gl == 0) & (gv == 1), (gl == 1) & (gv == 0), (gl == 1) & (gv == 1)
    cte = np.sum(w * (gammaln(gl + 1) + gammaln(gv + 1)))

    def f(p):
        mu, h, rho = p[0], p[1], p[2]
        at = p[3:3 + n] - p[3:3 + n].mean(); de = p[3 + n:] - p[3 + n:].mean()
        lh = np.exp(mu + h * hf + at[hi] - de[ai]); la = np.exp(mu + at[ai] - de[hi])
        tau = np.ones_like(lh); dth = np.zeros_like(lh); dta = np.zeros_like(lh); dtr = np.zeros_like(lh)
        tau[m00] = 1 - lh[m00] * la[m00] * rho; dth[m00] = -la[m00] * rho; dta[m00] = -lh[m00] * rho; dtr[m00] = -lh[m00] * la[m00]
        tau[m01] = 1 + lh[m01] * rho; dth[m01] = rho; dtr[m01] = lh[m01]
        tau[m10] = 1 + la[m10] * rho; dta[m10] = rho; dtr[m10] = la[m10]
        tau[m11] = 1 - rho; dtr[m11] = -1
        tau = np.maximum(tau, 1e-10)
        ll = np.sum(w * (gl * np.log(lh) - lh + gv * np.log(la) - la + np.log(tau))) - cte
        gh_ = w * (gl - lh + lh * dth / tau); ga_ = w * (gv - la + la * dta / tau)
        g_at = np.bincount(hi, gh_, n) + np.bincount(ai, ga_, n)
        g_de = -np.bincount(ai, gh_, n) - np.bincount(hi, ga_, n)
        g_at = g_at - reg * at; g_de = g_de - reg * de
        g = np.r_[gh_.sum() + ga_.sum(), np.sum(gh_ * hf), np.sum(w * dtr / tau), g_at - g_at.mean(), g_de - g_de.mean()]
        obj = ll - 0.5 * reg * (np.sum(at ** 2) + np.sum(de ** 2))
        return -obj, -g

    x0 = np.r_[0.2, 0.25, -0.05, np.zeros(2 * n)]
    bounds = [(-2, 2), (-1, 1), (-0.25, 0.25)] + [(-4, 4)] * (2 * n)
    r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 2000})
    p = r.x; at = p[3:3 + n] - p[3:3 + n].mean(); de = p[3 + n:] - p[3 + n:].mean()
    return dict(mu=float(p[0]), home=float(p[1]), rho=float(p[2]), ataque=dict(zip(eq, at)), defensa=dict(zip(eq, de)),
                partidos=len(t), ok=bool(r.success))


def matriz(lh, la, rho, grid=GRID):
    M = np.outer(poisson.pmf(np.arange(grid), lh), poisson.pmf(np.arange(grid), la))
    M[0, 0] *= 1 - lh * la * rho; M[0, 1] *= 1 + lh * rho; M[1, 0] *= 1 + la * rho; M[1, 1] *= 1 - rho
    M = np.maximum(M, 0)
    return M / M.sum()


def predecir(m, loc, vis, neutral, el, ev, k_elo):
    a, d = m["ataque"], m["defensa"]
    lh = np.exp(m["mu"] + (0 if neutral else m["home"]) + a.get(loc, 0) - d.get(vis, 0) + k_elo * (el - ev) / 400)
    la = np.exp(m["mu"] + a.get(vis, 0) - d.get(loc, 0) + k_elo * (ev - el) / 400)
    M = matriz(lh, la, m["rho"])
    i, j = np.unravel_index(M.argmax(), M.shape)
    return np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum(), lh + la, f"{i}-{j}"


def metricas(P, y):
    P = np.asarray(P, float); y = np.asarray(y)
    return {"partidos": int(len(y)), "acierto": float((P.argmax(1) == y).mean()),
            "brier": float(((P - np.eye(3)[y]) ** 2).sum(1).mean()),
            "logloss": float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean())}


def mezclar(pm, pc, w):
    p = np.power(np.clip(pc, 1e-9, 1), w) * np.power(np.clip(pm, 1e-9, 1), 1 - w)
    return p / p.sum(-1, keepdims=True)


def forma(df, eq, n=5):
    g = df[(df.local == eq) | (df.visita == eq)].tail(n)
    out = []
    for r in g.itertuples():
        es_l = r.local == eq; gf, gc = (r.gl, r.gv) if es_l else (r.gv, r.gl)
        out.append(dict(fecha=r.fecha.strftime("%Y-%m-%d"), rival=r.visita if es_l else r.local,
                        cond="N" if r.neutral else ("L" if es_l else "V"), gf=int(gf), gc=int(gc),
                        res="G" if gf > gc else ("E" if gf == gc else "P")))
    return out


def procesar_dc(cfg):
    es_sel = cfg["fuente"] == "intl"
    aj = AJUSTES["selecciones" if es_sel else "clubes"]
    if cfg["fuente"] == "new":
        datos = cargar_liga_new(cfg)
    elif cfg["fuente"] == "main":
        datos = cargar_liga_main(cfg, aj["anios"])
    else:
        datos = cargar_intl()
    datos = datos.dropna(subset=["fecha", "gl", "gv"]).sort_values("fecha", kind="stable").reset_index(drop=True)
    datos = datos[datos.fecha <= HOY]
    datos[["gl", "gv"]] = datos[["gl", "gv"]].astype(int)
    if es_sel:
        datos = datos[datos.fecha >= HOY - pd.Timedelta(days=365 * 20)].reset_index(drop=True)
        el, ev, elo_actual = elo_selecciones(datos)
    else:
        el, ev, elo_actual = elo_clubes(datos)
    datos["elo_l"], datos["elo_v"] = el, ev
    min_p = 8 if es_sel else 1
    print(f"   {len(datos)} partidos, del {datos.fecha.min():%d/%m/%Y} al {datos.fecha.max():%d/%m/%Y}")

    # ---------- Prueba honesta ----------
    inicio = datos.fecha.max() - pd.Timedelta(days=DIAS_PRUEBA)
    prueba = datos[datos.fecha >= inicio].copy()
    prueba["bloque"] = prueba.fecha.dt.to_period(aj["periodo"]).dt.start_time
    prueba["y"] = np.select([prueba.gl > prueba.gv, prueba.gl == prueba.gv], [0, 1], 2)
    resultados = {}
    t0 = time.time()
    for xi in aj["xi"]:
        for reg in aj["reg"]:
            filas = []
            for blo, g in prueba.groupby("bloque"):
                m = ajustar(datos, blo, xi, reg, aj["anios"], min_p)
                for r in g.itertuples():
                    if r.local not in m["ataque"] or r.visita not in m["ataque"]:
                        continue
                    for k in aj["k_elo"]:
                        p1, px, p2, gol, marc = predecir(m, r.local, r.visita, r.neutral, r.elo_l, r.elo_v, k)
                        filas.append((r.Index, k, p1, px, p2, gol, marc))
            resultados[(xi, reg)] = pd.DataFrame(filas, columns=["idx", "k", "p1", "px", "p2", "goles", "marcador"])
    print(f"   prueba: {len(prueba)} partidos, {time.time() - t0:.0f} s")

    tabla = []
    for (xi, reg), f in resultados.items():
        for k, g in f.groupby("k"):
            g = g.set_index("idx"); yy = prueba.loc[g.index, "y"].values
            tabla.append(dict(xi=xi, reg=reg, k=k, **metricas(g[["p1", "px", "p2"]].values, yy)))
    tabla = pd.DataFrame(tabla).sort_values("logloss")
    print(tabla.head(5).round(3).to_string(index=False))
    best = tabla.iloc[0]; XI, REG, K = float(best.xi), float(best.reg), float(best.k)
    bdf = resultados[(XI, REG)]; bdf = bdf[bdf.k == K].set_index("idx")
    pr = prueba.loc[bdf.index]; yv = pr.y.values; PM = bdf[["p1", "px", "p2"]].values
    comp = {"Modelo": metricas(PM, yv)}
    w_mercado = None
    con = pr[["casa_1", "casa_x", "casa_2"]].notna().all(axis=1).values
    if con.sum() > 30:
        PC = pr[["casa_1", "casa_x", "casa_2"]].values[con]
        comp = {"Modelo": metricas(PM[con], yv[con]), "Casas de apuestas": metricas(PC, yv[con])}
        pesos = [(w, metricas(mezclar(PM[con], PC, w), yv[con])["logloss"]) for w in np.round(np.arange(0, 1.01, 0.1), 1)]
        w_mercado = float(min(pesos, key=lambda x: x[1])[0])
        comp["Modelo + momios"] = metricas(mezclar(PM[con], PC, w_mercado), yv[con])
    comp["Siempre local"] = {"partidos": int(len(yv)), "acierto": float((yv == 0).mean()), "brier": None, "logloss": None}
    exacto = float(np.mean(bdf.marcador.values == (pr.gl.astype(str) + "-" + pr.gv.astype(str)).values))
    print(f"   mejor: xi={XI} reg={REG} k_elo={K} · peso momios={w_mercado} · acierto={comp['Modelo']['acierto']:.3f}")

    # ---------- Modelo final ----------
    final = ajustar(datos, HOY + pd.Timedelta(days=1), XI, REG, aj["anios"], min_p)
    activos = set(datos[datos.fecha >= HOY - pd.Timedelta(days=aj["dias_activo"])][["local", "visita"]].values.ravel())
    equipos = {e: {"ataque": round(float(final["ataque"][e]), 5), "defensa": round(float(final["defensa"][e]), 5),
                   "elo": round(float(elo_actual.get(e, 1500)), 1), "forma": forma(datos, e)}
               for e in sorted(final["ataque"]) if e in activos}
    return dict(
        tipo="dc", datos_hasta=datos.fecha.max().strftime("%Y-%m-%d"),
        params=dict(mu=final["mu"], home=final["home"], rho=final["rho"], k_elo=K, xi=XI, reg=REG, grid=GRID, w_mercado=w_mercado),
        equipos=equipos, prueba=comp, marcador_exacto=exacto, proximos=proximos(cfg, equipos),
        _datos=datos,
    )


# ============================== CHAMPIONS (ClubElo) ==============================
def calibrar_elo(conjuntos):
    """Relación entre diferencia de Elo y goles, aprendida con los partidos de ligas de los últimos 3 años."""
    d = pd.concat(conjuntos, ignore_index=True)
    d = d[d.fecha >= HOY - pd.Timedelta(days=3 * 365)]
    dif = (d.elo_l - d.elo_v).values / 400; gl, gv = d.gl.values, d.gv.values

    def nll(p):
        c0, h, c1, rho = p
        lh = np.exp(c0 + h + c1 * dif); la = np.exp(c0 - c1 * dif)
        tau = np.ones_like(lh)
        m = (gl == 0) & (gv == 0); tau[m] = 1 - lh[m] * la[m] * rho
        m = (gl == 0) & (gv == 1); tau[m] = 1 + lh[m] * rho
        m = (gl == 1) & (gv == 0); tau[m] = 1 + la[m] * rho
        m = (gl == 1) & (gv == 1); tau[m] = 1 - rho
        return -np.sum(poisson.logpmf(gl, lh) + poisson.logpmf(gv, la) + np.log(np.maximum(tau, 1e-10)))

    r = minimize(nll, [0.2, 0.25, 0.5, -0.05], method="L-BFGS-B", bounds=[(-1, 1), (-1, 1), (0, 3), (-0.25, 0.25)])
    c0, h, c1, rho = map(float, r.x)
    print(f"   calibración Elo→goles con {len(d)} partidos: c0={c0:.3f} local={h:.3f} c1={c1:.3f} rho={rho:.3f}")
    return c0, h, c1, rho


def procesar_champions(conjuntos):
    c0, h, c1, rho = calibrar_elo(conjuntos)
    raw, ultimo_error = None, None
    for dias_atras in range(4):                       # si hoy falla, prueba con días anteriores
        fecha = (HOY - pd.Timedelta(days=dias_atras)).strftime("%Y-%m-%d")
        for url in (f"http://api.clubelo.com/{fecha}", f"https://api.clubelo.com/{fecha}"):
            for intento in range(3):                  # el servidor de ClubElo a veces responde 502
                try:
                    raw = leer_csv(url)
                    if len(raw) > 50:
                        break
                except Exception as e:
                    ultimo_error = e
                    if not os.environ.get("PRUEBA_DIR"):
                        time.sleep(15)
            if raw is not None and len(raw) > 50:
                break
        if raw is not None and len(raw) > 50:
            break
    if raw is None or len(raw) <= 50:
        raise RuntimeError(f"ClubElo no respondió: {ultimo_error}")
    print(f"   ClubElo: ratings del {fecha}")
    club, pais, elo = col(raw, "Club", "club", "Team"), col(raw, "Country", "country"), col(raw, "Elo", "elo")
    raw = raw.dropna(subset=[club, elo]).sort_values(elo, ascending=False).head(250)
    equipos = {str(r[club]).strip(): {"elo": round(float(r[elo]), 1), "pais": str(r[pais]).strip()} for _, r in raw.iterrows()}
    return dict(tipo="elo", datos_hasta=fecha,
                params=dict(c0=c0, home=h, c1=c1, rho=rho, grid=GRID, w_mercado=None),
                equipos=equipos, prueba={}, proximos=[],
                nota="Ratings de ClubElo.com (todos los clubes de Europa en la misma escala). Sin prueba propia: no hay historial de Champions en la fuente.")


# ============================== PRINCIPAL ==============================
def guardar(cfg, res):
    salida = {"id": cfg["id"], "liga": cfg["nombre"], "bandera": cfg["bandera"],
              "generado": pd.Timestamp.now(tz="America/Mexico_City").strftime("%Y-%m-%d %H:%M")}
    salida.update({k: v for k, v in res.items() if not k.startswith("_")})
    with open(os.path.join(SALIDA, cfg["id"] + ".json"), "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, separators=(",", ":"))


def main():
    os.makedirs(SALIDA, exist_ok=True)
    indice_path = os.path.join(SALIDA, "index.json")
    try:
        previo = {l["id"]: l for l in json.load(open(indice_path, encoding="utf-8")).get("ligas", [])}
    except Exception:
        previo = {}
    indice, conjuntos, fallas = [], [], 0
    for cfg in LIGAS:
        print(f"\n=== {cfg['nombre']} ===")
        ent = {"id": cfg["id"], "nombre": cfg["nombre"], "bandera": cfg["bandera"], "archivo": cfg["id"] + ".json"}
        try:
            if cfg["tipo"] == "dc":
                res = procesar_dc(cfg)
                if cfg["fuente"] != "intl":
                    conjuntos.append(res["_datos"][["fecha", "gl", "gv", "elo_l", "elo_v"]])
            else:
                if not conjuntos:
                    raise RuntimeError("Champions necesita al menos una liga procesada")
                res = procesar_champions(conjuntos)
            guardar(cfg, res)
            ent.update(ok=True, actualizado=pd.Timestamp.now(tz="America/Mexico_City").strftime("%Y-%m-%d %H:%M"))
            print(f"   ✅ {len(res['equipos'])} equipos · {len(res['proximos'])} próximos partidos")
        except Exception as e:
            fallas += 1
            traceback.print_exc()
            viejo = previo.get(cfg["id"], {})
            existe = os.path.exists(os.path.join(SALIDA, ent["archivo"]))
            ent.update(ok=existe, error=str(e)[:200], actualizado=viejo.get("actualizado"))
            print(f"   ❌ {e}  (se conserva la versión anterior: {existe})")
        indice.append(ent)
    with open(indice_path, "w", encoding="utf-8") as f:
        json.dump({"ligas": indice, "actualizado": pd.Timestamp.now(tz="America/Mexico_City").strftime("%Y-%m-%d %H:%M")},
                  f, ensure_ascii=False, indent=1)
    print(f"\nTerminado: {len(LIGAS) - fallas} de {len(LIGAS)} competencias actualizadas.")
    if fallas == len(LIGAS):
        sys.exit(1)


if __name__ == "__main__":
    main()
