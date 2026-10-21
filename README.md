# Predictor de fútbol

App de pronósticos que se actualiza sola todos los días: Liga MX, Selecciones, LaLiga,
Premier League, Serie A, Primeira Liga, Eredivisie y Champions.

- `modelo.py` — descarga resultados y momios, entrena el modelo de cada competencia y escribe `docs/data/`.
- `docs/index.html` — la app (GitHub Pages la publica).
- `.github/workflows/actualizar.yml` — corre el modelo cada día a las 6:17 am (CDMX).

Correr a mano: pestaña **Actions → Actualizar pronósticos → Run workflow**.

Fuentes: football-data.co.uk (ligas y momios), martj42/international_results (selecciones), ClubElo.com (Champions).
