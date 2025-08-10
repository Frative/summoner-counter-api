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

def get_counters_for_champion(champion_name, top_n=None):
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
  model, champion_encoder, scaler, matchup_df, numeric_cols = load_model_and_objects()
  if champion_name not in champion_encoder.classes_:
    raise ValueError(f"'{champion_name}' no está en el codificador.")
  champ_index = champion_encoder.transform([champion_name])[0]
  champions = champion_encoder.classes_
  results = []
  for i, other_champ in enumerate(champions):
    if i == champ_index:
      continue
    row = matchup_df[
      (matchup_df['champ_a_encoded'] == i) &
      (matchup_df['champ_b_encoded'] == champ_index)
    ]
    if row.empty:
      continue
    features = row[['champ_a_encoded', 'champ_b_encoded'] + numeric_cols].copy()
    features[numeric_cols] = scaler.transform(features[numeric_cols])
    with torch.no_grad():
      inputs = torch.tensor(features.values, dtype=torch.float32).to(device)
      pred = model(inputs).item()
    results.append((other_champ, pred))
  results.sort(key=lambda x: x[1], reverse=True)
  return results if top_n is None else results[:top_n]