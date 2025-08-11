import torch
import joblib
import pandas as pd
import torch.nn as nn
import os

MODEL_FILE = "counter_model.pth"
ENCODER_FILE = "champion_encoder.joblib"
SCALER_FILE = "scaler.joblib"
DATA_SUMMARY_FILE = "matchup_summary.csv"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class CounterPredictor(nn.Module):
    def __init__(self, num_champions, input_features):
        super(CounterPredictor, self).__init__()

        # Capas de embedding para los campeones
        self.embedding_a = nn.Embedding(num_champions, 16)
        self.embedding_b = nn.Embedding(num_champions, 16)

        # Capas densas
        self.fc1 = nn.Linear(input_features + 32, 64)  # 32 por los embeddings concatenados
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Separar características
        champ_a = x[:, 0].long()
        champ_b = x[:, 1].long()
        numeric_features = x[:, 2:]

        # Obtener embeddings
        emb_a = self.embedding_a(champ_a)
        emb_b = self.embedding_b(champ_b)

        # Concatenar características
        x = torch.cat([emb_a, emb_b, numeric_features], dim=1)

        # Pasar por capas densas
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