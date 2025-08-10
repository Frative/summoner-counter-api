import torch
import joblib
import pandas as pd
import os

MODEL_FILE = "counter_model.pth"
ENCODER_FILE = "champion_encoder.joblib"
SCALER_FILE = "scaler.joblib"
DATA_SUMMARY_FILE = "matchup_summary.csv"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class CounterPredictor(torch.nn.Module):
  def __init__(self, num_champions, input_features):
    super().__init__()
    self.embedding_a = torch.nn.Embedding(num_champions, 16)
    self.embedding_b = torch.nn.Embedding(num_champions, 16)
    self.fc1 = torch.nn.Linear(input_features + 32, 64)
    self.fc2 = torch.nn.Linear(64, 32)
    self.fc3 = torch.nn.Linear(32, 16)
    self.output = torch.nn.Linear(16, 1)
    self.dropout = torch.nn.Dropout(0.2)
    self.relu = torch.nn.ReLU()
    self.sigmoid = torch.nn.Sigmoid()

  def forward(self, x):
    champ_a = x[:, 0].long()
    champ_b = x[:, 1].long()
    numeric_features = x[:, 2:]
    emb_a = self.embedding_a(champ_a)
    emb_b = self.embedding_b(champ_b)
    x = torch.cat([emb_a, emb_b, numeric_features], dim=1)
    x = self.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.relu(self.fc2(x))
    x = self.dropout(x)
    x = self.relu(self.fc3(x))
    x = self.output(x)
    x = self.sigmoid(x)
    return x

def load_model_and_objects():
  champion_encoder = joblib.load(ENCODER_FILE)
  scaler = joblib.load(SCALER_FILE)
  matchup_df = pd.read_csv(DATA_SUMMARY_FILE)
  num_champions = len(champion_encoder.classes_)
  numeric_cols = ['avg_phys_dmg_a', 'avg_magic_dmg_a', 'avg_true_dmg_a', 'avg_level_a',
          'avg_phys_dmg_b', 'avg_magic_dmg_b', 'avg_true_dmg_b', 'avg_level_b']
  input_features = len(numeric_cols)
  model = CounterPredictor(num_champions, input_features).to(device)
  model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
  model.eval()
  return model, champion_encoder, scaler, matchup_df, numeric_cols

def predict_team_matchup(team_a, team_b):
  """
  team_a: list of 5 champion names
  team_b: list of 5 champion names
  Returns: 5x5 matrix of predictions for each (a, b) pair
  """
  model, champion_encoder, scaler, matchup_df, numeric_cols = load_model_and_objects()
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
  inputs = torch.tensor(features_df.values, dtype=torch.float32).to(device)

  with torch.no_grad():
    preds = model(inputs).cpu().numpy().flatten()

  # Convertir a matriz 5x5
  matrix = preds.reshape(5, 5)
  return matrix