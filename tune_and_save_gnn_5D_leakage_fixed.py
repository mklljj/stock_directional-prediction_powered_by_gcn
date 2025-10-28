# tune_and_save_gnn_5D_leakage_fixed.py
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score
import optuna
import os
from tqdm import tqdm

# --- GNN Model Definition ---
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_layers, dropout_rate):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_node_features, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = x.relu()
            if i < len(self.convs) - 1:
                x = self.dropout(x)
        x = self.classifier(x)
        return x

# --- Manual Indicator Functions ---
def add_rsi(df, period=14):
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-9)
    df[f'RSI_{period}'] = 100 - (100 / (1 + rs))

def add_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    df['MACD_12_26_9'] = ema_fast - ema_slow
    df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=signal, adjust=False).mean()

def add_bollinger_bands(df, length=20, std=2):
    middle_band = df['close'].rolling(window=length).mean()
    std_dev = df['close'].rolling(window=length).std()
    upper_band = middle_band + (std_dev * std)
    lower_band = middle_band - (std_dev * std)
    denominator = (upper_band - lower_band).replace(0, 1e-9)
    df[f'BBP_{length}_{std}.0'] = (df['close'] - lower_band) / denominator

# --- Objective Function for Optuna ---
def objective(trial, data):
    hidden_channels = trial.suggest_categorical('hidden_channels', [16, 32, 64, 128])
    num_layers = trial.suggest_int('num_layers', 2, 4)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.6)

    model = GCN(
        num_node_features=data.num_node_features,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout_rate=dropout_rate
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_logits = model(data.x, data.edge_index)
        pred = pred_logits.argmax(dim=1)
        y_test = data.y[data.test_mask].numpy()
        y_pred = pred[data.test_mask].numpy()
        accuracy = accuracy_score(y_test, y_pred)

    return accuracy

# --- Main Script Execution ---
if __name__ == "__main__":
    # --- Load Full Dataset ---
    filename = 'stock_data_10years_20251023.csv'
    try:
        df_full = pd.read_csv(filename, parse_dates=['date'], index_col='date')
    except FileNotFoundError:
        print(f"❌ Error: The file '{filename}' was not found.")
        exit()

    df_full.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)

    unique_tickers = df_full['ticker'].unique()
    all_best_results = []
    min_records_threshold = 252 * 2 # Require 2 years of data
    N_TRIALS = 50

    models_dir = 'tuned_gnn_models_5D'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"✅ Created directory: {models_dir}")

    print(f"🚀 Starting 5-DAY AHEAD hyperparameter tuning for {len(unique_tickers)} tickers...")
    print(f"Each ticker will be tuned for {N_TRIALS} trials.")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # --- Loop Through Each Ticker ---
    for ticker in tqdm(unique_tickers, desc="Tuning and Saving 5D Tickers"):
        df = df_full[df_full['ticker'] == ticker].copy()

        if len(df) < min_records_threshold:
            continue

        try:
            # --- DATA LEAKAGE FIX STARTS HERE ---

            # 1. Define the target variable first.
            df['Future_Return_5D'] = df['close'].shift(-5) / df['close'] - 1
            # Drop rows where the future return is unknown (the last 5 rows).
            df.dropna(subset=['Future_Return_5D'], inplace=True)

            # 2. Split the DataFrame into training and testing sets BEFORE feature engineering.
            split_index = int(len(df) * 0.8)
            train_df = df.iloc[:split_index].copy()
            test_df = df.iloc[split_index:].copy()

            # 3. Apply feature engineering SEPARATELY to each set.
            for dataset in [train_df, test_df]:
                dataset['Return'] = dataset['close'].pct_change()
                add_rsi(dataset)
                add_macd(dataset)
                add_bollinger_bands(dataset)

            # 4. Drop any NaNs created by the indicators from each set.
            train_df.dropna(inplace=True)
            test_df.dropna(inplace=True)

            if train_df.empty or test_df.empty:
                tqdm.write(f" -> Skipping {ticker} due to insufficient data after processing.")
                continue

            # 5. Combine the processed dataframes to build the graph.
            # The split is maintained using masks, so no data leakage occurs here.
            combined_df = pd.concat([train_df, test_df])
            y_series = (combined_df['Future_Return_5D'] > 0).astype(int)
            X_df = combined_df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'ticker', 'Return', 'Future_Return_5D'])

            x = torch.tensor(X_df.values, dtype=torch.float)
            y = torch.tensor(y_series.values, dtype=torch.long)
            num_nodes = len(combined_df)
            edge_index = torch.stack([torch.arange(0, num_nodes - 1), torch.arange(1, num_nodes)], dim=0)

            graph_data = Data(x=x, edge_index=edge_index, y=y)

            # 6. Create train/test masks based on the lengths of the processed dataframes.
            num_train_nodes = len(train_df)
            graph_data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            graph_data.train_mask[:num_train_nodes] = True
            graph_data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            graph_data.test_mask[num_train_nodes:] = True

            # --- DATA LEAKAGE FIX ENDS HERE ---

            if graph_data.test_mask.sum() == 0:
                continue

            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: objective(trial, graph_data), n_trials=N_TRIALS, show_progress_bar=False)

            best_params = study.best_params
            best_model = GCN(
                num_node_features=graph_data.num_node_features,
                hidden_channels=best_params['hidden_channels'],
                num_layers=best_params['num_layers'],
                dropout_rate=best_params['dropout_rate']
            )
            optimizer = torch.optim.Adam(
                best_model.parameters(),
                lr=best_params['lr'],
                weight_decay=best_params['weight_decay']
            )
            criterion = torch.nn.CrossEntropyLoss()

            best_model.train()
            for epoch in range(200):
                optimizer.zero_grad()
                out = best_model(graph_data.x, graph_data.edge_index)
                loss = criterion(out[graph_data.train_mask], graph_data.y[graph_data.train_mask])
                loss.backward()
                optimizer.step()

            model_path = os.path.join(models_dir, f'{ticker}_best_gnn_5D.pth')
            torch.save(best_model.state_dict(), model_path)

            report_params = study.best_params
            report_params['ticker'] = ticker
            report_params['best_accuracy'] = study.best_value * 100
            all_best_results.append(report_params)

        except Exception as e:
            tqdm.write(f" -> ⚠️ Skipping {ticker} due to a critical error: {e}")
            continue

    # --- Display Final Summary ---
    print("\n\n--- 5-Day Ahead Hyperparameter Tuning Final Summary ---")
    if all_best_results:
        results_df = pd.DataFrame(all_best_results)
        cols = ['ticker', 'best_accuracy'] + [col for col in results_df.columns if col not in ['ticker', 'best_accuracy']]
        results_df = results_df[cols]
        results_df.sort_values(by='best_accuracy', ascending=False, inplace=True)
        print(results_df.to_string(index=False, float_format="%.4f"))
    else:
        print("No models were successfully tuned.")
