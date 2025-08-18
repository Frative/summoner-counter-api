# Counter Predictor Model

Este proyecto implementa un modelo de *machine learning* en PyTorch para predecir los **counters entre campeones** en base a datos de partidas.

## Dataset
El dataset utilizado contiene información de interacciones entre campeones, incluyendo estadísticas de daño y nivel en asesinatos y muertes.

Enlace al dataset: [Mega.nz](https://mega.nz/file/8QxRyIIL#-cjAbjleYh_k6_pwaVrsC9Wp2nI7Dxe9BuDtgQzoac0)

## Flujo del sistema

1. **Preprocesamiento (`preprocess_and_save`)**
   - Carga el dataset crudo (`datos.csv`).
   - Calcula métricas de enfrentamientos entre campeones (kills, deaths, win_ratio).
   - Genera pares de campeones con estadísticas promedio.
   - Aplica *LabelEncoder* para codificar campeones.
   - Normaliza variables numéricas con *StandardScaler*.
   - Guarda:
     - `champion_encoder.joblib`
     - `scaler.joblib`
     - `matchup_summary.csv`

2. **Definición del modelo (`CounterPredictor`)**
   - Dos *embeddings* de tamaño 16 para campeones A y B.
   - Concatenación de embeddings + características numéricas.
   - Red neuronal totalmente conectada:
     - FC1: 64 neuronas
     - FC2: 32 neuronas
     - FC3: 16 neuronas
     - Salida: 1 neurona con *sigmoid* (probabilidad de victoria).
   - Regularización con *Dropout*.

3. **Entrenamiento (`train_and_save_model`)**
   - Entrena usando *BCELoss* (función de entropía cruzada binaria).
   - Optimización con *Adam*.
   - Incluye pérdida de consistencia (symmetry loss) para asegurar que si A gana a B, entonces B pierde contra A.
   - Guarda el modelo entrenado en `counter_model.pth`.

4. **Inferencia**
   - `generate_counter_matrix`: genera matriz de counters entre todos los campeones.
   - `get_counters_for_champion`: dado un campeón, devuelve la lista de campeones que lo contrarrestan con mayor probabilidad.

## Archivos principales
- `datos.csv`: dataset base.
- `champion_encoder.joblib`: codificador de campeones.
- `scaler.joblib`: escalador de características numéricas.
- `matchup_summary.csv`: dataset procesado.
- `counter_model.pth`: modelo entrenado.

## Requisitos
```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch joblib
```

## Uso
Ejecutar directamente:
```bash
python main.py
```
Esto entrenará el modelo (si no existe previamente) y mostrará los counters de un campeón de ejemplo.

---

### Ejemplo de salida
```
AurelionSol: Ezreal 0.73
AurelionSol: Zed 0.65
...

