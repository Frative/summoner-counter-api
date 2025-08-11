import torch
import joblib
import pandas as pd
import torch.nn as nn
import os

import model

def get_counters_for_champion(champion_name, top_n=None, threshold = 0):
  """
  Este endpoint especifica qué tan probable es que `champion_name` muera por los campeones retornados.

  Args:
    champion_name (str): Nombre del campeón para el cual se buscan los counters.
    top_n (int, optional): Número máximo de campeones a retornar. Si es None, retorna todos.

  Returns:
    list of tuples: Lista de tuplas (nombre_del_campeón, probabilidad) ordenadas por probabilidad descendente.

  Raises:
    ValueError: Si el nombre del campeón no está en el codificador.
  """
  _model, champion_encoder, scaler, matchup_df, numeric_cols = model.load_model_and_objects()
  if champion_name not in champion_encoder.classes_:
      raise ValueError(f"'{champion_name}' no está en el codificador.")

  champ_index = champion_encoder.transform([champion_name])[0]
  champions = champion_encoder.classes_
  results = []

  _model.eval()
  with torch.no_grad():
      for i, other_champ in enumerate(champions):
          if i == champ_index:
              continue

          # Probabilidad de other_champ vs champion_name
          row = matchup_df[
              (matchup_df['champ_a_encoded'] == i) &
              (matchup_df['champ_b_encoded'] == champ_index)
          ]
          if row.empty:
              continue

          features = row[['champ_a_encoded', 'champ_b_encoded'] + numeric_cols].copy()
          features[numeric_cols] = scaler.transform(features[numeric_cols])
          inputs = torch.tensor(features.values, dtype=torch.float32).to(model.device)
          pred = _model(inputs).item()

          # Probabilidad inversa: champion_name vs other_champ
          inverse_row = matchup_df[
              (matchup_df['champ_a_encoded'] == champ_index) &
              (matchup_df['champ_b_encoded'] == i)
          ]
          if inverse_row.empty:
              continue

          inverse_features = inverse_row[['champ_a_encoded', 'champ_b_encoded'] + numeric_cols].copy()
          inverse_features[numeric_cols] = scaler.transform(inverse_features[numeric_cols])
          inverse_inputs = torch.tensor(inverse_features.values, dtype=torch.float32).to(model.device)
          inverse_pred = _model(inverse_inputs).item()

          # Solo incluir si pred es alta y inverse_pred es baja
          if pred > threshold and inverse_pred < (1 - threshold):
              results.append((other_champ, pred))

  results.sort(key=lambda x: x[1], reverse=True)
  return results if top_n is None else results[:top_n]