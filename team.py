import torch
import joblib
import pandas as pd
import torch.nn as nn
import os

import model

def predict_team_matchup(team_a, team_b):
  """
  team_a: list of 5 champion names
  team_b: list of 5 champion names
  Returns: 5x5 matrix of predictions for each (a, b) pair
  """
  _model, champion_encoder, scaler, matchup_df, numeric_cols = model.load_model_and_objects()
  if len(team_a) != 5 or len(team_b) != 5:
    raise ValueError("Ambos equipos deben tener exactamente 5 campeones.")
  for champ in team_a + team_b:
    if champ not in champion_encoder.classes_:
      raise ValueError(f"'{champ}' no está en el codificador.")

  team_a_indices = champion_encoder.transform(team_a)
  team_b_indices = champion_encoder.transform(team_b)

  features_list = []
  for i, a_idx in enumerate(team_a_indices):
    for j, b_idx in enumerate(team_b_indices):
      row = matchup_df[
        (matchup_df['champ_a_encoded'] == a_idx) &
        (matchup_df['champ_b_encoded'] == b_idx)
      ]
      if row.empty:
        # Si no hay datos, usar ceros en las features
        numeric = [0.0] * len(numeric_cols)
      else:
        numeric = row[numeric_cols].values[0]
      features = [a_idx, b_idx] + list(numeric)
      features_list.append(features)

  # Normalizar las features numéricas
  features_df = pd.DataFrame(features_list, columns=['champ_a_encoded', 'champ_b_encoded'] + numeric_cols)
  features_df[numeric_cols] = scaler.transform(features_df[numeric_cols])
  inputs = torch.tensor(features_df.values, dtype=torch.float32).to(model.device)

  with torch.no_grad():
    preds = _model(inputs).cpu().numpy().flatten()

  # Convertir a matriz 5x5
  matrix = preds.reshape(5, 5)
  return matrix