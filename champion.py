import torch
import pandas as pd

import model  # importa el módulo de arriba (mismo paquete/proyecto)

@torch.no_grad()
def _build_feature_row(df_sum: pd.DataFrame, scaler, numeric_cols, champ_a: str, champ_b: str,
                       a_idx: int, b_idx: int) -> pd.DataFrame:
    r = df_sum[(df_sum['champ_a']==champ_a) & (df_sum['champ_b']==champ_b)].copy()
    if r.empty:
        feats = {k:0.0 for k in numeric_cols}
        r = pd.DataFrame([{**feats, 'champ_a_encoded':a_idx, 'champ_b_encoded':b_idx}])
    else:
        # Si hay múltiples filas del par, promediar features
        r = r.groupby(['champ_a','champ_b'], as_index=False)[numeric_cols].mean()
        r[numeric_cols] = scaler.transform(r[numeric_cols])
        r['champ_a_encoded'] = a_idx
        r['champ_b_encoded'] = b_idx
    r = r[['champ_a_encoded','champ_b_encoded'] + numeric_cols]
    return r

@torch.no_grad()
def get_counters_for_champion(champion_name: str, top_n: int | None = None, threshold: float = 0.0):
    """
    Devuelve campeones que más probablemente maten a `champion_name`.
    Retorna lista de (campeón, probabilidad) ordenada descendentemente.
    Probabilidad calculada como promedio de: P(other > champion_name) y 1 - P(champion_name > other).
    """
    _model, champion_encoder, scaler, matchup_df, numeric_cols = model.load_model_and_objects()

    if champion_name not in champion_encoder.classes_:
        raise ValueError(f"'{champion_name}' no está en el codificador.")

    champ_index = champion_encoder.transform([champion_name])[0]
    champions = champion_encoder.classes_
    results: list[tuple[str, float]] = []

    for i, other_champ in enumerate(champions):
        if i == champ_index:
            continue
        # Índices codificados
        other_idx = i

        # other vs target
        row_ab = _build_feature_row(matchup_df, scaler, numeric_cols,
                                    other_champ, champion_name, other_idx, champ_index)
        x_ab = torch.tensor(row_ab.values, dtype=torch.float32, device=model.device)
        p_ab = torch.sigmoid(_model(x_ab)).mean().item()

        # target vs other
        row_ba = _build_feature_row(matchup_df, scaler, numeric_cols,
                                    champion_name, other_champ, champ_index, other_idx)
        x_ba = torch.tensor(row_ba.values, dtype=torch.float32, device=model.device)
        p_ba = torch.sigmoid(_model(x_ba)).mean().item()

        # Enforce antisymmetry at inference
        prob = 0.5 * (p_ab + (1.0 - p_ba))

        if prob > threshold:
            results.append((other_champ, prob))

    results.sort(key=lambda x: x[1], reverse=True)
    return results if top_n is None else results[:top_n]
