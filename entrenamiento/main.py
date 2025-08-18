import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import joblib

# Visualizar matriz
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Archivos para guardar objetos reutilizables
ENCODER_FILE = "champion_encoder.joblib"
SCALER_FILE = "scaler.joblib"
DATA_SUMMARY_FILE = "matchup_summary.csv"
MODEL_FILE = "counter_model.pth"

# Prepare dataset for the model
def prepare_matchup_data(matchups, data):
    # Asegurar nombres consistentes
    matchups = matchups.rename(columns={
        "killer_champion_x": "champ_a",
        "victim_champion_x": "champ_b"
    })

    # 1. Calcular estadísticas de kills (champ_a)
    kills_stats = (
        data.groupby(["killer_champion", "victim_champion"])
        .agg(
            avg_phys_dmg_a=("killer_damage_physical", "mean"),
            avg_magic_dmg_a=("killer_damage_magic", "mean"),
            avg_true_dmg_a=("killer_damage_true", "mean"),
            avg_level_a=("killer_level", "mean"),
        )
        .reset_index()
        .rename(columns={
            "killer_champion": "champ_a",
            "victim_champion": "champ_b"
        })
    )

    # 2. Calcular estadísticas de deaths (champ_b)
    deaths_stats = (
        data.groupby(["victim_champion", "killer_champion"])
        .agg(
            avg_phys_dmg_b=("killer_damage_physical", "mean"),
            avg_magic_dmg_b=("killer_damage_magic", "mean"),
            avg_true_dmg_b=("victim_damage_true", "mean"),
            avg_level_b=("victim_level", "mean"),
        )
        .reset_index()
        .rename(columns={
            "victim_champion": "champ_a",
            "killer_champion": "champ_b"
        })
    )

    # 3. Merge de todo
    result = (
        matchups
        .merge(kills_stats, on=["champ_a", "champ_b"], how="left")
        .merge(deaths_stats, on=["champ_a", "champ_b"], how="left")
    )

    # 4. Rellenar NaN con 0
    result.fillna(0, inplace=True)

    return result



class ChampionMatchupDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Definir modelo
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

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=50, consistency_weight=0.1):
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Pérdida de consistencia
            inverse_inputs = inputs.clone()
            inverse_inputs[:, [0, 1]] = inverse_inputs[:, [1, 0]]  # Intercambiar campeones
            inverse_inputs[:, 2:] = inputs[:, 2:][:, [4, 5, 6, 7, 0, 1, 2, 3]]  # Intercambiar características numéricas
            inverse_outputs = model(inverse_inputs)
            consistency_loss = torch.mean((outputs + inverse_outputs - 1) ** 2)
            total_loss = loss + consistency_weight * consistency_loss

            total_loss.backward()
            optimizer.step()
            running_loss += loss.item()  # Reportar solo la pérdida principal

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

        test_loss = test_loss / len(test_loader)
        test_losses.append(test_loss)

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    return train_losses, test_losses

def generate_counter_matrix(model, champion_encoder, scaler, numeric_cols):
    champions = champion_encoder.classes_
    num_champions = len(champions)

    counter_matrix = np.zeros((num_champions, num_champions))

    # Crear datos para todos los pares posibles
    for i, champ_a in enumerate(champions):
        for j, champ_b in enumerate(champions):
            if i == j:
                counter_matrix[i, j] = 0.5  # Empate contra sí mismo
                continue

            # Obtener estadísticas promedio para este par
            champ_a_data = data[data['killer_champion'] == champ_a]
            champ_b_data = data[data['killer_champion'] == champ_b]

            # Crear características
            features = {
                'champ_a_encoded': i,
                'champ_b_encoded': j,
                'avg_phys_dmg_a': champ_a_data['killer_damage_physical'].mean() if not champ_a_data.empty else 0,
                'avg_magic_dmg_a': champ_a_data['killer_damage_magic'].mean() if not champ_a_data.empty else 0,
                'avg_true_dmg_a': champ_a_data['killer_damage_true'].mean() if not champ_a_data.empty else 0,
                'avg_level_a': champ_a_data['killer_level'].mean() if not champ_a_data.empty else 0,
                'avg_phys_dmg_b': champ_b_data['killer_damage_physical'].mean() if not champ_b_data.empty else 0,
                'avg_magic_dmg_b': champ_b_data['killer_damage_magic'].mean() if not champ_b_data.empty else 0,
                'avg_true_dmg_b': champ_b_data['killer_damage_true'].mean() if not champ_b_data.empty else 0,
                'avg_level_b': champ_b_data['killer_level'].mean() if not champ_b_data.empty else 0
            }

            # Convertir a DataFrame y escalar
            features_df = pd.DataFrame([features])
            features_df[numeric_cols] = scaler.transform(features_df[numeric_cols])

            # Predecir
            with torch.no_grad():
                inputs = torch.tensor(features_df.values, dtype=torch.float32).to(device)
                pred = model(inputs).item()


            counter_matrix[i, j] = pred

    return champions, counter_matrix

def get_counters_for_champion(
    model,
    champion_name,
    matchup_df,
    champion_encoder,
    scaler,
    numeric_cols,
    top_n=None,
    threshold=0
):
    if champion_name not in champion_encoder.classes_:
        raise ValueError(f"'{champion_name}' no está en el codificador.")

    champ_index = champion_encoder.transform([champion_name])[0]
    champions = champion_encoder.classes_
    results = []

    model.eval()
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
            inputs = torch.tensor(features.values, dtype=torch.float32).to(device)
            pred = model(inputs).item()

            # Probabilidad inversa: champion_name vs other_champ
            inverse_row = matchup_df[
                (matchup_df['champ_a_encoded'] == champ_index) &
                (matchup_df['champ_b_encoded'] == i)
            ]
            if inverse_row.empty:
                continue

            inverse_features = inverse_row[['champ_a_encoded', 'champ_b_encoded'] + numeric_cols].copy()
            inverse_features[numeric_cols] = scaler.transform(inverse_features[numeric_cols])
            inverse_inputs = torch.tensor(inverse_features.values, dtype=torch.float32).to(device)
            inverse_pred = model(inverse_inputs).item()

            # Solo incluir si pred es alta y inverse_pred es baja
            if pred > threshold and inverse_pred < (1 - threshold):
                results.append((other_champ, pred))

    results.sort(key=lambda x: x[1], reverse=True)
    return results if top_n is None else results[:top_n]


def preprocess_and_save():
    print("Procesando datos...")
    data = pd.read_csv("datos.csv")

    # Crear variable objetivo: ratio de victorias por par de campeones
    matchups = data.groupby(['killer_champion', 'victim_champion']).size().reset_index(name='kills')
    reverse_matchups = data.groupby(['victim_champion', 'killer_champion']).size().reset_index(name='deaths')

    # Merge con nombres claros
    matchups = matchups.merge(reverse_matchups,
                             left_on=['killer_champion', 'victim_champion'],
                             right_on=['victim_champion', 'killer_champion'],
                             how='outer').fillna(0)

    # Renombrar columnas para consistencia
    matchups = matchups.rename(columns={
        'killer_champion_x': 'champ_a',
        'victim_champion_x': 'champ_b',
        'killer_champion_y': 'killer_champion_reverse',
        'victim_champion_y': 'victim_champion_reverse'
    })

    # Eliminar columnas redundantes si existen
    matchups = matchups.drop(columns=['killer_champion_reverse', 'victim_champion_reverse'], errors='ignore')

    matchups['total_encounters'] = matchups['kills'] + matchups['deaths']
    matchups['win_ratio'] = matchups['kills'] / matchups['total_encounters']

    # Filtrar pares con suficientes enfrentamientos
    matchups = matchups[matchups['total_encounters'] >= 3]

    # Crear pares inversos explícitamente
    inverse_matchups = matchups.copy()
    inverse_matchups['champ_a'], inverse_matchups['champ_b'] = (
        matchups['champ_b'], matchups['champ_a']
    )
    inverse_matchups['kills'], inverse_matchups['deaths'] = (
        matchups['deaths'], matchups['kills']
    )
    inverse_matchups['win_ratio'] = 1 - matchups['win_ratio']

    # Combinar ambos DataFrames
    matchups = pd.concat([matchups, inverse_matchups], ignore_index=True)
    matchups = matchups.groupby(['champ_a', 'champ_b']).mean().reset_index()

    # Codificar campeones
    champion_encoder = LabelEncoder()
    all_champions = pd.concat([data['killer_champion'], data['victim_champion']]).unique()
    champion_encoder.fit(all_champions)

    matchup_df = prepare_matchup_data(matchups, data)
    matchup_df = matchup_df[(matchup_df['champ_a'] != 0) & (matchup_df['champ_b'] != 0)]

    matchup_df['champ_a_encoded'] = champion_encoder.transform(matchup_df['champ_a'])
    matchup_df['champ_b_encoded'] = champion_encoder.transform(matchup_df['champ_b'])

    numeric_cols = ['avg_phys_dmg_a', 'avg_magic_dmg_a', 'avg_true_dmg_a', 'avg_level_a',
                    'avg_phys_dmg_b', 'avg_magic_dmg_b', 'avg_true_dmg_b', 'avg_level_b']
    scaler = StandardScaler()
    matchup_df[numeric_cols] = scaler.fit_transform(matchup_df[numeric_cols])

    # Guardar objetos
    joblib.dump(champion_encoder, ENCODER_FILE)
    joblib.dump(scaler, SCALER_FILE)
    matchup_df.to_csv(DATA_SUMMARY_FILE, index=False)

    print("Datos preprocesados y guardados.")

def load_model_and_objects():
    if not os.path.exists(MODEL_FILE):
        print("No se encontró el modelo entrenado, entrenando primero...")
        train_and_save_model()

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

def train_and_save_model():
    if not (os.path.exists(ENCODER_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(DATA_SUMMARY_FILE)):
        print("Faltan archivos preprocesados, ejecutando preprocesamiento...")
        preprocess_and_save()

    champion_encoder = joblib.load(ENCODER_FILE)
    scaler = joblib.load(SCALER_FILE)
    matchup_df = pd.read_csv(DATA_SUMMARY_FILE)

    numeric_cols = ['avg_phys_dmg_a', 'avg_magic_dmg_a', 'avg_true_dmg_a', 'avg_level_a',
                    'avg_phys_dmg_b', 'avg_magic_dmg_b', 'avg_true_dmg_b', 'avg_level_b']

    X = matchup_df[['champ_a_encoded', 'champ_b_encoded'] + numeric_cols]
    y = matchup_df['win_ratio']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = ChampionMatchupDataset(X_train, y_train)
    test_dataset = ChampionMatchupDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    num_champions = len(champion_encoder.classes_)
    input_features = X_train.shape[1] - 2

    model = CounterPredictor(num_champions, input_features).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses, test_losses = train_model(model, train_loader, test_loader, criterion, optimizer, epochs=50)

    torch.save(model.state_dict(), MODEL_FILE)
    print(f"Modelo guardado como '{MODEL_FILE}'")

if __name__ == "__main__":
    model, champion_encoder, scaler, matchup_df, numeric_cols = load_model_and_objects()

    counters = get_counters_for_champion(
        model=model,
        champion_name='AurelionSol',
        matchup_df=matchup_df,
        champion_encoder=champion_encoder,
        scaler=scaler,
        numeric_cols=numeric_cols,
        top_n=None  # TODOS los counters
    )

    for champ, prob in counters:
        print(f"{champ}: {prob:.2f}")
