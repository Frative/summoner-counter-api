import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import model

class Evaluator:
    def __init__(self):
        self._model, self.champion_encoder, self.scaler, self.matchup_df, self.numeric_cols = model.load_model_and_objects()

    def plot_confusion_matrix(self, threshold=0.5, output_file="confusion_matrix.png"):
        # Features y etiquetas
        X = self.matchup_df[['champ_a_encoded', 'champ_b_encoded'] + self.numeric_cols].copy()

        # Convertir win_ratio a etiquetas binarias
        y_true = (self.matchup_df['win_ratio'] > 0.5).astype(int).values  

        # Normalizar
        X[self.numeric_cols] = self.scaler.transform(X[self.numeric_cols])
        inputs = torch.tensor(X.values, dtype=torch.float32).to(model.device)

        # Predicciones
        with torch.no_grad():
            y_pred_probs = self._model(inputs).cpu().numpy().flatten()

        y_pred = (y_pred_probs > threshold).astype(int)

        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(cmap="Blues")
        plt.title("Matriz de Confusión del Modelo")

        # Guardar como imagen
        plt.savefig(output_file)
        plt.close()
        print(f"✅ Matriz de confusión guardada en {output_file}")


if __name__ == "__main__":
    e = Evaluator()
    e.plot_confusion_matrix(output_file="confusion_matrix.png")
