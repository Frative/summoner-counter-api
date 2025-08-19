#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import model

@torch.no_grad()
def evaluate(threshold=0.5, antisymmetry=True, min_encounters=1, include_ties=False, save_csv=None, plot=False, batch_size=4096):
    
    _model, champion_encoder, scaler, matchup_df, numeric_cols = model.load_model_and_objects()
    _model.eval()


    df = matchup_df.copy()
    # excluir espejos perfectos (mismo campeón contra sí mismo)
    df = df[df['champ_a'] != df['champ_b']]
    # mínimo de partidas
    df = df[df['total_encounters'].astype(float) >= float(min_encounters)]
    # omitir empates si no se piden
    if not include_ties:
        df = df[np.abs(df['win_ratio'].astype(float) - 0.5) > 1e-6]

    # Si no hay datos tras el filtro, salir temprano
    if df.empty:
        print("Sin pares válidos tras el filtrado.")
        return

    y_true = (df['win_ratio'].astype(float).values > 0.5).astype(int)


    # AB (A vs B)
    a_enc = df['champ_a_encoded'].astype(np.int64).values
    b_enc = df['champ_b_encoded'].astype(np.int64).values

    num_ab = df[numeric_cols].astype(float).fillna(0.0).values
    num_ab = scaler.transform(num_ab)

    X_ab = np.concatenate([
        a_enc.reshape(-1,1).astype(np.float32),
        b_enc.reshape(-1,1).astype(np.float32),
        num_ab.astype(np.float32)
    ], axis=1)


    def infer_in_batches(X: np.ndarray) -> np.ndarray:
        probs = np.empty((X.shape[0],), dtype=np.float32)
        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])
            xb = torch.tensor(X[start:end], dtype=torch.float32, device=model.device)
            pb = torch.sigmoid(_model(xb)).squeeze(1).detach().cpu().numpy().astype(np.float32)
            probs[start:end] = pb
        return probs

    p_ab = infer_in_batches(X_ab)


    if antisymmetry:
        # Construir BA agregando a un solo registro por par (champ_b, champ_a)
        rev = matchup_df[['champ_a_encoded','champ_b_encoded'] + numeric_cols].copy()
        # Después de renombrar, cada fila representa (B,A)
        rev = rev.rename(columns={'champ_a_encoded':'champ_b_encoded', 'champ_b_encoded':'champ_a_encoded'})
        # Promediar numéricos por par BA para evitar duplicados que rompen la dimensionalidad
        rev_grouped = (
            rev.groupby(['champ_a_encoded','champ_b_encoded'], as_index=False)[numeric_cols]
               .mean()
        )
        merged = df[['champ_a_encoded','champ_b_encoded']].merge(
            rev_grouped,
            on=['champ_a_encoded','champ_b_encoded'],
            how='left'
        )
        num_ba = merged[numeric_cols].astype(float).fillna(0.0).values
        num_ba = scaler.transform(num_ba)

        X_ba = np.concatenate([
            b_enc.reshape(-1,1).astype(np.float32),
            a_enc.reshape(-1,1).astype(np.float32),
            num_ba.astype(np.float32)
        ], axis=1)

        p_ba = infer_in_batches(X_ba)
        p = 0.5 * (p_ab + (1.0 - p_ba))
    else:
        p = p_ab


    y_pred = (p >= float(threshold)).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    print("Matriz de Confusión:")
    print(cm)
    print("\nReporte de Clasificación:")
    print(classification_report(y_true, y_pred, digits=3))


    if save_csv:
        out = df[['champ_a','champ_b']].copy()
        out['y_true'] = y_true
        out['p_pred'] = p
        out['y_pred'] = y_pred
        out.to_csv(save_csv, index=False)
        print(f"Predicciones guardadas en {save_csv}")

  
    if plot:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Neg","Pos"])
        disp.plot(cmap="Blues", values_format="d")
        plt.title("Matriz de Confusión")
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.5, help='Umbral de clasificación para p(A>B)')
    parser.add_argument('--min_encounters', type=int, default=3, help='Mínimo de partidas para considerar un par')
    parser.add_argument('--include_ties', action='store_true', help='Incluir casos con win_ratio==0.5 como y=0')
    parser.add_argument('--antisymmetry', action='store_true', help='Forzar antisimetría en inferencia')
    parser.add_argument('--save_csv', default='', help='Ruta opcional para guardar predicciones detalladas')
    parser.add_argument('--plot', action='store_true', help='Mostrar gráfico de matriz de confusión')
    parser.add_argument('--batch_size', type=int, default=4096, help='Tamaño de lote para inferencia vectorizada')
    args = parser.parse_args()

    evaluate(threshold=args.threshold,
             antisymmetry=args.antisymmetry,
             min_encounters=args.min_encounters,
             include_ties=args.include_ties,
             save_csv=args.save_csv if args.save_csv else None,
             plot=args.plot,
             batch_size=args.batch_size)
