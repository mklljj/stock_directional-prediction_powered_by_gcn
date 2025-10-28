# tune_and_save_gnn.py
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

# --- GNN Model Definition (Same as before) ---
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

# --- Manual Indicator Functions (Same as before) ---
def add_rsi(df, period=14):
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
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
    df[f'BBP_{length}_{std}.0'] = (df['close'] - lower_band) / (upper_band - lower_band)

# --- Objective Function for Optuna (Same as before) ---
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
    filename = 'stock_data_10years_20251028.csv'
    try:
        df_full = pd.read_csv(filename, parse_dates=['date'], index_col='date')
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        exit()

    df_full.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)

    # --- Get Tickers and Prepare for Loop ---
    unique_tickers = df_full['ticker'].unique()
    all_best_results = []
    min_records_threshold = 252
    N_TRIALS = 25

    # --- NEW: Create a directory to save the tuned models ---
    models_dir = 'tuned_gnn_models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created directory: {models_dir}")

    print(f"Starting hyperparameter tuning for {len(unique_tickers)} tickers...")
    print(f"Each ticker will be tuned for {N_TRIALS} trials.")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # --- Loop Through Each Ticker ---
    for ticker in tqdm(unique_tickers, desc="Tuning and Saving Tickers"):
        df = df_full[df_full['ticker'] == ticker].copy()

        if len(df) < min_records_threshold:
            continue

        try:
            # --- Feature Engineering ---
            df['Return'] = df['close'].pct_change()
            df['Future_Return'] = df['Return'].shift(-1)
            add_rsi(df); add_macd(df); add_bollinger_bands(df)
            df['interaction_vix_rsi'] = df['^VIX_return'] * df['RSI_14']
            df['interaction_gspc_bbp'] = df['^GSPC_return'] * df['BBP_20_2.0']
            df.dropna(inplace=True)

            if df.empty or len(df) < min_records_threshold:
                continue

            # --- Convert Data to Graph Structure ---
            y_series = (df['Future_Return'] > 0).astype(int)
            X_df = df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'ticker', 'Return', 'Future_Return'])
            x = torch.tensor(X_df.values, dtype=torch.float)
            y = torch.tensor(y_series.values, dtype=torch.long)
            num_nodes = len(df)
            edge_index = torch.stack([torch.arange(0, num_nodes - 1), torch.arange(1, num_nodes)], dim=0)
            graph_data = Data(x=x, edge_index=edge_index, y=y)

            # --- Create Train/Test Masks ---
            split_index = int(num_nodes * 0.8)
            graph_data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            graph_data.train_mask[:split_index] = True
            graph_data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            graph_data.test_mask[split_index:] = True

            if graph_data.test_mask.sum() == 0:
                continue

            # --- Run the Optuna Study ---
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: objective(trial, graph_data), n_trials=N_TRIALS, show_progress_bar=False)

            # --- NEW SECTION: Retrain and Save the Best Model ---
            best_params = study.best_params

            # 1. Initialize the best model with optimal hyperparameters
            best_model = GCN(
                num_node_features=graph_data.num_node_features,
                hidden_channels=best_params['hidden_channels'],
                num_layers=best_params['num_layers'],
                dropout_rate=best_params['dropout_rate']
            )

            # 2. Set up the optimizer with optimal learning rate and weight decay
            optimizer = torch.optim.Adam(
                best_model.parameters(),
                lr=best_params['lr'],
                weight_decay=best_params['weight_decay']
            )
            criterion = torch.nn.CrossEntropyLoss()

            # 3. Retrain this final model on the training data
            best_model.train()
            for epoch in range(200): # Use the same number of epochs
                optimizer.zero_grad()
                out = best_model(graph_data.x, graph_data.edge_index)
                loss = criterion(out[graph_data.train_mask], graph_data.y[graph_data.train_mask])
                loss.backward()
                optimizer.step()

            # 4. Save the trained model's state dictionary
            model_path = os.path.join(models_dir, f'{ticker}_best_gnn.pth')
            torch.save(best_model.state_dict(), model_path)
            # --- END OF NEW SECTION ---

            # --- Store the results for the final report (same as before) ---
            report_params = study.best_params
            report_params['ticker'] = ticker
            report_params['best_accuracy'] = study.best_value * 100
            all_best_results.append(report_params)

        except Exception as e:
            tqdm.write(f"  -> Skipping {ticker} due to a critical error: {e}")
            continue

    # --- Display Final Summary ---
    print("\n\n--- Hyperparameter Tuning Final Summary ---")
    if all_best_results:
        results_df = pd.DataFrame(all_best_results)
        cols = ['ticker', 'best_accuracy'] + [col for col in results_df.columns if col not in ['ticker', 'best_accuracy']]
        results_df = results_df[cols]
        results_df.sort_values(by='best_accuracy', ascending=False, inplace=True)
        print(results_df.to_string(index=False, float_format="%.4f"))
    else:
        print("No models were successfully tuned.")
