import os
import joblib
import pandas as pd
import torch
import torch.nn as nn

MODEL_FILE = "counter_model.pth"
ENCODER_FILE = "champion_encoder.joblib"
SCALER_FILE = "scaler.joblib"
DATA_SUMMARY_FILE = "matchup_summary.csv"

NUMERIC_COLS = [
    'avg_phys_dmg_a','avg_magic_dmg_a','avg_true_dmg_a','avg_level_a',
    'avg_phys_dmg_b','avg_magic_dmg_b','avg_true_dmg_b','avg_level_b',
    'avg_assistants','avg_game_time'
]

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

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
        ca = x[:,0].long(); cb = x[:,1].long(); feats = x[:,2:]
        ea = self.embedding_a(ca); eb = self.embedding_b(cb)
        z = torch.cat([ea, eb, feats], dim=1)
        z = self.act(self.fc1(z)); z = self.dropout(z)
        z = self.act(self.fc2(z)); z = self.dropout(z)
        z = self.act(self.fc3(z))
        logits = self.out(z)          
        return logits


def load_model_and_objects():
    # Carga artefactos y construye el modelo con el número correcto de features
    encoder = joblib.load(ENCODER_FILE)
    scaler = joblib.load(SCALER_FILE)
    matchup_df = pd.read_csv(DATA_SUMMARY_FILE)

    num_champions = len(encoder.classes_)
    input_features = len(NUMERIC_COLS)

    model = CounterPredictor(num_champions, input_features).to(device)
    state = torch.load(MODEL_FILE, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model, encoder, scaler, matchup_df, NUMERIC_COLS