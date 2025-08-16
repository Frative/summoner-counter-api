# API de counters (FastAPI + PyTorch)

Servicio REST para estimar **matchups** entre campeones y listar **counters** con PyTorch y FastAPI. Este README prioriza el uso del **Makefile** para evitar comandos Docker directos.

## TL;DR (Makefile)

```bash
make build   # construye la imagen: summoner-counter-api
make run     # levanta en segundo plano en :8000
# Opcional
make shell   # abre bash en el contenedor (puerto 8000 mapeado)
make stop    # detiene el contenedor
```

---

## Makefile

Objetivos disponibles:

```makefile
build:  docker build -t summoner-counter-api .
run:    docker run -d -p 8000:8000 --name summoner-counter-api --restart unless-stopped summoner-counter-api
shell:  docker run --rm -it -p 8000:8000 summoner-counter-api /bin/bash
stop:   docker stop summoner-counter-api
```

**Notas**

- `run` usa `--restart unless-stopped` para resiliencia.
- `shell` no monta el código host. Para depurar archivos locales monta volúmenes según necesidad.
- Imagen y contenedor se llaman `summoner-counter-api`.

---

## Qué hace

- **GET /champion/{name}/counters**: lista campeones que contrarrestan a `name` con probabilidad.
- **GET /team/counters**: devuelve matriz **5×5** de probabilidades para cada par A vs B.

El modelo `CounterPredictor` usa embeddings (A y B de 16 dims), features numéricas agregadas y capas densas con sigmoide.

---

## Requisitos

- Archivos en el **root**:
  - `counter_model.pth`
  - `champion_encoder.joblib`
  - `scaler.joblib`
  - `matchup_summary.csv`
- Docker y Make disponibles. Alternativa local: Python 3.11 + pip.

> En contenedor: CPU. En macOS puede usar **MPS** si ejecutas local fuera de Docker.

---

## Estructura

```
.
├── Dockerfile
├── Makefile
├── requirements.txt
├── main.py
├── model.py
├── team.py
├── champion.py
├── counter_model.pth              # <— proveer
├── champion_encoder.joblib        # <— proveer
├── scaler.joblib                  # <— proveer
└── matchup_summary.csv            # <— proveer
```

---

## Alternativas de ejecución

### A) Con Makefile (recomendado)

```bash
make build
make run
# Logs (si lo necesitas)
docker logs -f summoner-counter-api
# Parar
make stop
```

### B) Docker directo (equivalente)

```bash
docker build -t summoner-counter-api .
docker run -d -p 8000:8000 --name summoner-counter-api --restart unless-stopped summoner-counter-api
```

### C) Local (sin Docker)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

---

## Endpoints

### 1) `GET /champion/{champion_name}/counters`

Ejemplo:

```bash
curl -s "http://localhost:8000/champion/Ahri/counters" | jq
```

Respuesta:

```json
{"counters": [["Zed", 0.73], ["Fizz", 0.69], ["Kassadin", 0.66]]}
```

Errores: `"'X' no está en el codificador."` si el campeón no existe en el encoder.

### 2) `GET /team/counters`

Query params (5 por equipo): `team_a`, `team_b`.

```bash
curl -s \
  "http://localhost:8000/team/counters?\
team_a=Ahri&team_a=Lee%20Sin&team_a=Garen&team_a=Lux&team_a=Jinx&\
team_b=Zed&team_b=Kha'Zix&team_b=Darius&team_b=Morgana&team_b=Caitlyn" | jq
```

Respuesta: matriz 5×5 donde `[i][j]` es **team\_a[i] vs team\_b[j]**. Errores: tamaño de equipo ≠ 5 o nombres fuera del encoder.

---

## Funcionamiento

1. `model.load_model_and_objects` carga encoder, scaler, CSV de matchups y pesos del modelo.
2. Features: índices de campeones + `avg_phys_dmg_*`, `avg_magic_dmg_*`, `avg_true_dmg_*`, `avg_level_*`.
3. Inferencia:
   - `team.predict_team_matchup` construye 25 pares, completa faltantes con ceros, normaliza y predice.
   - `champion.get_counters_for_champion` filtra counters con umbral, verificando la inversa.
