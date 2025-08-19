import torch
import pandas as pd
import numpy as np

import model

@torch.no_grad()
def _build_row(df_sum: pd.DataFrame, scaler, numeric_cols, champ_a: str, champ_b: str,
               a_idx: int, b_idx: int) -> pd.DataFrame:
    r = df_sum[(df_sum['champ_a']==champ_a) & (df_sum['champ_b']==champ_b)].copy()
    if r.empty:
        feats = {k:0.0 for k in numeric_cols}
        r = pd.DataFrame([{**feats, 'champ_a_encoded':a_idx, 'champ_b_encoded':b_idx}])
    else:
        r = r.groupby(['champ_a','champ_b'], as_index=False)[numeric_cols].mean()
        r[numeric_cols] = scaler.transform(r[numeric_cols])
        r['champ_a_encoded'] = a_idx
        r['champ_b_encoded'] = b_idx
    r = r[['champ_a_encoded','champ_b_encoded'] + numeric_cols]
    return r

@torch.no_grad()
def predict_team_matchup(team_a: list[str], team_b: list[str], antisymmetry: bool = True) -> np.ndarray:
    """
    team_a: lista de 5 campeones (A1..A5)
    team_b: lista de 5 campeones (B1..B5)
    Return: matriz 5x5 con P(Ai > Bj) para cada par.
    Si antisymmetry=True, devuelve 0.5*(P(Ai>Bj) + 1 - P(Bj>Ai)) por par.
    """
    _model, champion_encoder, scaler, matchup_df, numeric_cols = model.load_model_and_objects()

    if len(team_a) != 5 or len(team_b) != 5:
        raise ValueError("Ambos equipos deben tener exactamente 5 campeones.")

    for champ in team_a + team_b:
        if champ not in champion_encoder.classes_:
            raise ValueError(f"'{champ}' no está en el codificador.")

    a_idx = champion_encoder.transform(team_a)
    b_idx = champion_encoder.transform(team_b)

    rows_ab = []
    rows_ba = [] if antisymmetry else None

    # Construir todas las filas Ai vs Bj y opcionalmente Bj vs Ai
    for i, ca in enumerate(team_a):
        for j, cb in enumerate(team_b):
            ra = _build_row(matchup_df, scaler, numeric_cols, ca, cb, a_idx[i], b_idx[j])
            rows_ab.append(ra)
            if antisymmetry:
                rb = _build_row(matchup_df, scaler, numeric_cols, cb, ca, b_idx[j], a_idx[i])
                rows_ba.append(rb)

    X_ab = torch.tensor(pd.concat(rows_ab, ignore_index=True).values, dtype=torch.float32, device=model.device)
    logits_ab = _model(X_ab)
    p_ab = torch.sigmoid(logits_ab).cpu().numpy().reshape(5,5)

    if antisymmetry:
        X_ba = torch.tensor(pd.concat(rows_ba, ignore_index=True).values, dtype=torch.float32, device=model.device)
        logits_ba = _model(X_ba)
        p_ba = torch.sigmoid(logits_ba).cpu().numpy().reshape(5,5)
        return 0.5 * (p_ab + (1.0 - p_ba.T)) 
        return p_ab
