#!/usr/bin/env python3
"""
Modelo de probabilidad de victoria A vs B usando red neuronal profunda de 3 capas.
- Dataset: datos.csv con columnas exactas:
  killer_champion,killer_damage_physical,killer_damage_magic,killer_damage_true,killer_level,
  victim_champion,victim_damage_physical,victim_damage_magic,victim_damage_true,victim_level,
  assistants,game_time

Objetivo: P(A gana a B) en un 1v1 basado en métricas agregadas por par del dataset.

Diario del errante:
- Preprocesado explícito y reproducible (semillas, escalado estándar manual, codificación LabelEncoder guardada).
- Construcción de features por par (A como killer, B como víctima) + espejo inverso.
- Red MLP con 3 capas ocultas, inicialización Xavier explícita y ReLU.
- Entrenamiento con BCELogits estable + regularización de consistencia A_vs_B + B_vs_A -> 1.
- Bucle de entrenamiento manual con clipping de gradiente y registro por época.
- Guardado/ carga de artefactos: encoder, scaler, resumen de pares, pesos.
- Inferencia determinista: función y CLI para predecir P(A>B).

NOTA: Sin abstracciones de alto nivel; todo está escrito de forma explícita con PyTorch.
"""
from __future__ import annotations
import os
import argparse
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# ========= Config =========
SEED = 1337
ENCODER_FILE = "champion_encoder.joblib"
SCALER_FILE = "scaler.joblib"
DATA_SUMMARY_FILE = "matchup_summary.csv"
MODEL_FILE = "counter_model.pth"
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 60
CONSISTENCY_WEIGHT = 0.5
GRAD_CLIP_NORM = 1.0

# ========= Determinismo =========
def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# ========= Dispositivo =========
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# ========= Dataset helpers =========
NUMERIC_COLS = [
    'avg_phys_dmg_a','avg_magic_dmg_a','avg_true_dmg_a','avg_level_a',
    'avg_phys_dmg_b','avg_magic_dmg_b','avg_true_dmg_b','avg_level_b',
    'avg_assistants','avg_game_time'
]

def prepare_matchup_data(matchups: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """Enriquece pares (A,B) con medias de métricas cuando A mata a B y viceversa."""
    # Estadísticas cuando A es killer y B víctima
    kills_stats = (
        data.groupby(["killer_champion","victim_champion"], as_index=False)
            .agg(
                avg_phys_dmg_a=("killer_damage_physical","mean"),
                avg_magic_dmg_a=("killer_damage_magic","mean"),
                avg_true_dmg_a=("killer_damage_true","mean"),
                avg_level_a=("killer_level","mean"),
                avg_assistants=("assistants","mean"),
                avg_game_time=("game_time","mean"),
            )
            .rename(columns={"killer_champion":"champ_a","victim_champion":"champ_b"})
    )

    # Estadísticas del lado de B (cuando B fue killer y A víctima) para rellenar la mitad B
    deaths_stats = (
        data.groupby(["victim_champion","killer_champion"], as_index=False)
            .agg(
                avg_phys_dmg_b=("killer_damage_physical","mean"),  # daño recibido por A cuando B mató
                avg_magic_dmg_b=("killer_damage_magic","mean"),
                avg_true_dmg_b=("killer_damage_true","mean"),
                avg_level_b=("killer_level","mean"),
            )
            .rename(columns={"victim_champion":"champ_a","killer_champion":"champ_b"})
    )

    result = matchups.merge(kills_stats, on=["champ_a","champ_b"], how="left")\
                     .merge(deaths_stats, on=["champ_a","champ_b"], how="left")

    # NaN -> 0
    result[NUMERIC_COLS] = result[NUMERIC_COLS].fillna(0.0)
    return result

class ChampionMatchupDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).view(-1,1)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ========= Modelo (3 capas ocultas) =========
class CounterPredictor(nn.Module):
    def __init__(self, num_champions: int, input_features: int, emb_dim: int = 16):
        super().__init__()
        self.embedding_a = nn.Embedding(num_champions, emb_dim)
        self.embedding_b = nn.Embedding(num_champions, emb_dim)
        self.fc1 = nn.Linear(input_features + 2*emb_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.15)
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.05)
    def forward(self, x):
        champ_a = x[:,0].long()
        champ_b = x[:,1].long()
        feats = x[:,2:]
        ea = self.embedding_a(champ_a)
        eb = self.embedding_b(champ_b)
        z = torch.cat([ea, eb, feats], dim=1)
        z = self.act(self.fc1(z))
        z = self.dropout(z)
        z = self.act(self.fc2(z))
        z = self.dropout(z)
        z = self.act(self.fc3(z))
        logits = self.out(z)
        return logits  # usar sigmoid en inferencia

# ========= Preprocesado =========

def preprocess_and_save(csv_path: str = "datos.csv"):
    print("Procesando datos…")
    data = pd.read_csv(csv_path)

    # Conteos de enfrentamientos por par (A killer, B victim)
    kills = data.groupby(["killer_champion","victim_champion"], as_index=False).size()
    kills = kills.rename(columns={"killer_champion":"champ_a","victim_champion":"champ_b","size":"kills"})

    # Enfrentamientos inversos: B killer, A victim
    deaths = data.groupby(["victim_champion","killer_champion"], as_index=False).size()
    deaths = deaths.rename(columns={"victim_champion":"champ_a","killer_champion":"champ_b","size":"deaths"})

    # Merge y métricas objetivo
    matchups = kills.merge(deaths, on=["champ_a","champ_b"], how="outer").fillna(0)
    matchups["total_encounters"] = matchups["kills"] + matchups["deaths"]
    # Filtrar pares con señal suficiente
    matchups = matchups[matchups["total_encounters"] >= 3].copy()
    matchups["win_ratio"] = matchups["kills"] / matchups["total_encounters"]

    # Crear espejo explícito B vs A para balancear
    inv = matchups.copy()
    inv[["champ_a","champ_b"]] = inv[["champ_b","champ_a"]]
    inv[["kills","deaths"]] = inv[["deaths","kills"]]
    inv["win_ratio"] = 1.0 - inv["win_ratio"]
    matchups = pd.concat([matchups, inv], ignore_index=True)

    # Encoder de campeones a índices
    encoder = LabelEncoder()
    all_champs = pd.unique(pd.concat([data['killer_champion'], data['victim_champion']], ignore_index=True))
    encoder.fit(all_champs)

    # Enriquecer con features numéricos por par
    enriched = prepare_matchup_data(matchups, data)

    # Codificar A y B
    enriched['champ_a_encoded'] = encoder.transform(enriched['champ_a'])
    enriched['champ_b_encoded'] = encoder.transform(enriched['champ_b'])

    # Ajustar scaler pero NO transformar aquí para evitar reescalado doble en entrenamiento/inferencia
    scaler = StandardScaler()
    scaler.fit(enriched[NUMERIC_COLS])

    # Persistir artefactos
    enriched.to_csv(DATA_SUMMARY_FILE, index=False)
    joblib.dump(encoder, ENCODER_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print("Preprocesado OK.")

# ========= Entrenamiento =========

def train_and_save_model():
    if not (os.path.exists(DATA_SUMMARY_FILE) and os.path.exists(ENCODER_FILE) and os.path.exists(SCALER_FILE)):
        preprocess_and_save()

    df = pd.read_csv(DATA_SUMMARY_FILE)
    X = df[['champ_a_encoded','champ_b_encoded'] + NUMERIC_COLS].copy()
    y = df['win_ratio']

    # Escalado aquí usando el scaler guardado
    scaler = joblib.load(SCALER_FILE)
    X[NUMERIC_COLS] = scaler.transform(X[NUMERIC_COLS])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=None)

    ds_tr = ChampionMatchupDataset(X_train, y_train)
    ds_va = ChampionMatchupDataset(X_val, y_val)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    num_champions = joblib.load(ENCODER_FILE).classes_.shape[0]
    input_features = len(NUMERIC_COLS)
    model = CounterPredictor(num_champions, input_features).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS+1):
        model.train()
        ep_loss = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss_main = criterion(logits, yb)
            # Consistencia: P(A>B) + P(B>A) ≈ 1
            inv = xb.clone()
            # swap índices de campeones
            inv[:,[0,1]] = inv[:,[1,0]]
            # swap bloques numéricos A<->B
            a_block = inv[:,2:2+4]  # a_phys, a_magic, a_true, a_level (antes del swap)
            b_block = inv[:,2+4:2+8]  # b_phys, b_magic, b_true, b_level
            inv[:,2:2+4], inv[:,2+4:2+8] = b_block, a_block
            # assistants y game_time permanecen
            inv_logits = model(inv)
            loss_cons = torch.mean((torch.sigmoid(logits) + torch.sigmoid(inv_logits) - 1.0)**2)
            loss = loss_main + CONSISTENCY_WEIGHT * loss_cons
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optim.step()
            ep_loss += loss_main.item()
        train_losses.append(ep_loss/len(dl_tr))

        # Validación
        model.eval()
        with torch.no_grad():
            ev = 0.0
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                ev += criterion(logits, yb).item()
        val_losses.append(ev/len(dl_va))
        print(f"Epoch {epoch:03d} | Train {train_losses[-1]:.4f} | Val {val_losses[-1]:.4f}")

    torch.save(model.state_dict(), MODEL_FILE)
    print(f"Modelo guardado en {MODEL_FILE}")

# ========= Carga =========

def load_all():
    if not os.path.exists(MODEL_FILE):
        train_and_save_model()
    encoder = joblib.load(ENCODER_FILE)
    scaler = joblib.load(SCALER_FILE)
    df = pd.read_csv(DATA_SUMMARY_FILE)
    num_champions = len(encoder.classes_)
    model = CounterPredictor(num_champions, len(NUMERIC_COLS)).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()
    return model, encoder, scaler, df

# ========= Inferencia =========

def predict_proba(model: CounterPredictor, encoder: LabelEncoder, scaler: StandardScaler,
                  df_summary: pd.DataFrame, champ_a: str, champ_b: str) -> float:
    if champ_a not in encoder.classes_ or champ_b not in encoder.classes_:
        raise ValueError("Campeón desconocido para el encoder.")
    a_idx = encoder.transform([champ_a])[0]
    b_idx = encoder.transform([champ_b])[0]

    def _build_row(df_sum, ca, cb, ai, bi):
        r = df_sum[(df_sum['champ_a']==ca) & (df_sum['champ_b']==cb)].copy()
        if r.empty:
            feats = {k:0.0 for k in NUMERIC_COLS}
            r = pd.DataFrame([{**feats, 'champ_a_encoded':ai, 'champ_b_encoded':bi}])
        else:
            r = r.groupby(['champ_a','champ_b'], as_index=False)[NUMERIC_COLS].mean()
            r[NUMERIC_COLS] = scaler.transform(r[NUMERIC_COLS])
            r['champ_a_encoded'] = ai
            r['champ_b_encoded'] = bi
        r = r[['champ_a_encoded','champ_b_encoded'] + NUMERIC_COLS]
        return r

    row_ab = _build_row(df_summary, champ_a, champ_b, a_idx, b_idx)
    x_ab = torch.tensor(row_ab.values, dtype=torch.float32, device=device)
    with torch.no_grad():
        p_ab = torch.sigmoid(model(x_ab)).mean().item()

    # Inferencia antisimétrica opcional: usar también B vs A si existe
    row_ba = df_summary[(df_summary['champ_a']==champ_b) & (df_summary['champ_b']==champ_a)]
    if not row_ba.empty:
        row_ba = _build_row(df_summary, champ_b, champ_a, b_idx, a_idx)
        x_ba = torch.tensor(row_ba.values, dtype=torch.float32, device=device)
        with torch.no_grad():
            p_ba = torch.sigmoid(model(x_ba)).mean().item()
        p = 0.5*(p_ab + (1.0 - p_ba))
    else:
        p = p_ab
    return float(p)

# ========= CLI =========

def main():
    parser = argparse.ArgumentParser(description="Predicción P(A gana a B)")
    sub = parser.add_subparsers(dest='cmd', required=True)

    pprep = sub.add_parser('preprocess', help='Preprocesa datos y guarda artefactos')
    pprep.add_argument('--csv', default='datos.csv', help='Ruta al CSV de datos')

    ptrain = sub.add_parser('train', help='Entrena el modelo y guarda pesos')

    ppred = sub.add_parser('predict', help='Predice probabilidad de victoria A sobre B')
    ppred.add_argument('--a', required=True, help='Nombre campeón A')
    ppred.add_argument('--b', required=True, help='Nombre campeón B')

    args = parser.parse_args()

    if args.cmd == 'preprocess':
        preprocess_and_save(args.csv)
    elif args.cmd == 'train':
        train_and_save_model()
    elif args.cmd == 'predict':
        model, encoder, scaler, df = load_all()
        p = predict_proba(model, encoder, scaler, df, args.a, args.b)
        print(f"P({args.a} gana a {args.b}) = {p:.4f}")

if __name__ == '__main__':
    main()
