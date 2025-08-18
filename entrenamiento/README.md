# Modelo de probabilidad de victoria A vs B (PyTorch)

Predice $P(A > B)$ en duelos 1v1 usando un MLP con embeddings por campeón y regularización de **consistencia antisimétrica** ($P(A>B)+P(B>A)\approx1$). Sin frameworks de alto nivel.

---

## Dataset
El dataset utilizado contiene información de interacciones entre campeones, incluyendo estadísticas de daño y nivel en asesinatos y muertes.

Enlace al dataset: [Mega.nz](https://mega.nz/file/8QxRyIIL#-cjAbjleYh_k6_pwaVrsC9Wp2nI7Dxe9BuDtgQzoac0)

## Tabla de contenidos

* [Idea](#idea)
* [Esquema de datos](#esquema-de-datos)
* [Pipeline](#pipeline)
* [Ingeniería de *features*](#ingenieria-de-features)
* [Arquitectura](#arquitectura)
* [Función de pérdida](#funcion-de-perdida)
* [Entrenamiento](#entrenamiento)
* [Determinismo y reproducibilidad](#determinismo-y-reproducibilidad)
* [Artefactos](#artefactos)
* [CLI](#cli)
* [Inferencia programática](#inferencia-programatica)
* [Validaciones y métricas sugeridas](#validaciones-y-metricas-sugeridas)
* [Suposiciones y límites](#suposiciones-y-limites)
* [Requisitos](#requisitos)
* [Estructura de archivos](#estructura-de-archivos)

---

## Idea

A partir de históricos de asesinatos entre campeones (A *killer* de B, y el espejo B de A), se entrena un clasificador binario que devuelve la probabilidad de que A gane sobre B dado el contexto medio observado.

## Esquema de datos

Archivo `datos.csv` con columnas exactas:

```
killer_champion,killer_damage_physical,killer_damage_magic,killer_damage_true,killer_level,
victim_champion,victim_damage_physical,victim_damage_magic,victim_damage_true,victim_level,
assistants,game_time
```

Cada fila es un evento A mata a B con métricas en ese instante.

## Pipeline

1. **Agregación por par (A,B)**: conteos `kills` y `deaths` (espejo), `total_encounters` y `win_ratio = kills/total`.
2. **Filtrado**: se conservan pares con `total_encounters >= 3` para señal mínima.
3. **Espejo explícito**: se añade (B,A) con `win_ratio = 1 - win_ratio`.
4. **Enriquecimiento numérico** por par con medias condicionadas:

   * cuando A fue *killer* de B → bloque **A**;
   * cuando B fue *killer* de A → bloque **B**.
5. **Codificación** de campeones con `LabelEncoder` persistido.
6. **Escalado** `StandardScaler` ajustado solo sobre columnas numéricas y persistido.
7. **Partición** `train/val`.

## Ingeniería de *features*

Columnas numéricas usadas (`NUMERIC_COLS`):

* A: `avg_phys_dmg_a`, `avg_magic_dmg_a`, `avg_true_dmg_a`, `avg_level_a`
* B: `avg_phys_dmg_b`, `avg_magic_dmg_b`, `avg_true_dmg_b`, `avg_level_b`
* Partida: `avg_assistants`, `avg_game_time`

Entrada final al MLP: `[champ_a_encoded, champ_b_encoded] + z`, donde `z` es el vector escalado de 10 *features*.

## Arquitectura

**CounterPredictor**

* Dos `nn.Embedding(num_champions, emb_dim=16)` independientes para A y B.
* Bloques densos: `Linear(d_in→64) → ReLU → Dropout(0.15) → Linear(64→32) → ReLU → Dropout(0.15) → Linear(32→16) → ReLU → Linear(16→1)`.
* **Inicialización**: `Xavier (Glorot) uniforme` en `Linear`, ceros en *bias*, `Embedding` normal $\mu=0,\,\sigma=0.05$.
* Salida: *logits*; la sigmoide se aplica solo en evaluación.

## Función de pérdida

* **Principal**: `BCEWithLogitsLoss` sobre $y\in[0,1]$ con *target* `win_ratio`.
* **Consistencia antisimétrica**: para un batch `x`, se crea `inv` invirtiendo (A,B) y los bloques A↔B. Penalización:
  $\mathcal{L}_{cons} = \mathbb{E}\big[(\sigma(f(x)) + \sigma(f(inv)) - 1)^2\big]$
* **Total**: $\mathcal{L}=\mathcal{L}_{BCE}+\lambda\,\mathcal{L}_{cons}$, con `λ = CONSISTENCY_WEIGHT = 0.5`.

## Entrenamiento

* Optimizador: `Adam(lr=1e-3)`.
* Tamaño de batch: `64`.
* Épocas: `60`.
* *Gradient clipping*: norma máx. `1.0`.
* *Dataloaders* con *shuffle* en entrenamiento.
* Registro por época de pérdida *train* y *val*.

## Determinismo y reproducibilidad

* Semillas fijas (`1337`) para NumPy y PyTorch.
* CUDA/MPS en modo determinista (`cudnn.deterministic=True`, `benchmark=False`).
* Persistencia de artefactos: encoder, scaler, resumen y pesos.

## Artefactos

* `champion_encoder.joblib` — `LabelEncoder` ajustado.
* `scaler.joblib` — `StandardScaler` ajustado a `NUMERIC_COLS`.
* `matchup_summary.csv` — dataset agregado y enriquecido por par.
* `counter_model.pth` — pesos del MLP.

## CLI

```bash
# 1) Preprocesar
python model.py preprocess --csv datos.csv

# 2) Entrenar
python model.py train

# 3) Predecir prob. de victoria
python model.py predict --a "CampeonA" --b "CampeonB"
# Salida: P(CampeonA gana a CampeonB) = 0.73xx
```

## Inferencia programática

```python
model, enc, scaler, df = load_all()
p = predict_proba(model, enc, scaler, df, "CampeonA", "CampeonB")
print(p)  # float en [0,1]
```

La inferencia aplica combinación opcional con el espejo si existe: $\hat p = 0.5\,[\sigma(f(A,B)) + (1-\sigma(f(B,A)))]$.

## Validaciones y métricas sugeridas

* **AUC/ROC** y **Brier score** en validación.
* **Calibración** (reliability diagrams) y **consistencia** $\sigma(f(A,B))+\sigma(f(B,A))\approx1$.
* *Ablations*: sin embeddings, sin consistencia, sin bloques A/B.

## Suposiciones y límites

* El *target* `win_ratio` aproxima $P(A>B)$ bajo medias históricas. No capta composición de equipo ni *items*.
* Pocas observaciones por par pueden sesgar. Se filtra `>=3`, pero evaluar sensibilidad.
* **Cero-imputación** en *features* cuando falta historial del par.

## Requisitos

* Python 3.10+
* PyTorch, scikit-learn, pandas, numpy, joblib

## Estructura de archivos

```
.
├── datos.csv
├── model.py                 # código principal (este repositorio)
├── champion_encoder.joblib  # generado
├── scaler.joblib            # generado
├── matchup_summary.csv      # generado
└── counter_model.pth        # generado
```

---

### Notas técnicas

* Dispositivo: `cuda` si disponible, luego `mps`, si no `cpu`.
* *Embeddings* separados para A y B para capturar roles no conmutativos.
* Escalado aplicado **solo** en entrenamiento e inferencia, no durante `fit` de scaler, evitando fuga de información.
* *Dropout* moderado para evitar sobreajuste en capas medias.
* *Stratify=None*: el *target* es continuo en \[0,1].
* La regularización empuja la antisimetría sin imponerla de forma rígida.
