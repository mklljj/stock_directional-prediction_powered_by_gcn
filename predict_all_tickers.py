# predict_all_tickers.py
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import os
from tqdm import tqdm # For a nice progress bar

# --- GNN Model Definition (MUST BE IDENTICAL to the training script) ---
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

# --- Manual Indicator Functions (MUST BE IDENTICAL to the training script) ---
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

# --- Main Batch Prediction Script ---
if __name__ == "__main__":
    # --- File and Directory Paths ---
    hyperparams_file = 'tuned_gnn_hyperparameters.csv'
    data_file = 'stock_data_10years_20251023.csv'
    models_dir = 'tuned_gnn_models'
    output_file = 'gnn_all_predictions.csv'

    # 1. Load Hyperparameters and Full Dataset
    try:
        df_params = pd.read_csv(hyperparams_file)
        df_full = pd.read_csv(data_file, parse_dates=['date'], index_col='date')
        df_full.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    except FileNotFoundError as e:
        print(f"❌ Error: A required file was not found. Please check the path.\n{e}")
        exit()

    all_predictions = []

    print(f"Starting batch prediction for {len(df_params)} tickers...")

    # 2. Loop Through Each Ticker in the Hyperparameters File
    for index, params in tqdm(df_params.iterrows(), total=df_params.shape[0], desc="Predicting Tickers"):
        ticker_symbol = params['ticker']

        try:
            # Prepare data for the specific ticker
            df = df_full[df_full['ticker'] == ticker_symbol].copy()
            if df.empty:
                continue

            # Apply feature engineering
            df['Return'] = df['close'].pct_change()
            df['Future_Return'] = df['Return'].shift(-1)
            add_rsi(df); add_macd(df); add_bollinger_bands(df)
            df['interaction_vix_rsi'] = df['^VIX_return'] * df['RSI_14']
            df['interaction_gspc_bbp'] = df['^GSPC_return'] * df['BBP_20_2.0']
            df_processed = df.dropna(subset=df.columns.difference(['Future_Return']))

            if df_processed.empty:
                continue

            last_day_date = df_processed.index[-1].strftime('%Y-%m-%d')

            # Build the graph
            X_df = df_processed.drop(columns=['open', 'high', 'low', 'close', 'volume', 'ticker', 'Return', 'Future_Return'])
            x = torch.tensor(X_df.values, dtype=torch.float)
            num_nodes = len(X_df)
            edge_index = torch.stack([torch.arange(0, num_nodes - 1), torch.arange(1, num_nodes)], dim=0)
            graph_data = Data(x=x, edge_index=edge_index)

            # Initialize Model and Load Trained Weights
            model = GCN(
                num_node_features=graph_data.num_node_features,
                hidden_channels=int(params['hidden_channels']),
                num_layers=int(params['num_layers']),
                dropout_rate=params['dropout_rate']
            )
            model_path = os.path.join(models_dir, f'{ticker_symbol}_best_gnn.pth')
            model.load_state_dict(torch.load(model_path))
            model.eval()

            # Make Prediction
            with torch.no_grad():
                last_day_logits = model(graph_data.x, graph_data.edge_index)[-1]
                probabilities = torch.softmax(last_day_logits, dim=0)
                prediction_index = probabilities.argmax().item()
                confidence = probabilities.max().item() * 100

            prediction_map = {0: "DOWN", 1: "UP"}

            # Store result
            all_predictions.append({
                'ticker': ticker_symbol,
                'last_data_date': last_day_date,
                'prediction': prediction_map[prediction_index],
                'confidence_percent': confidence
            })

        except Exception as e:
            # This ensures that if one ticker fails, the whole script doesn't stop
            tqdm.write(f"⚠️  Skipped {ticker_symbol} due to an error: {e}")
            continue

    # 3. Save All Predictions to a CSV File
    if all_predictions:
        df_results = pd.DataFrame(all_predictions)
        df_results.sort_values(by='confidence_percent', ascending=False, inplace=True)
        df_results.to_csv(output_file, index=False)
        print(f"\n✅ Predictions complete! Results for {len(df_results)} tickers saved to '{output_file}'")
    else:
        print("\n🤷 No predictions were generated. Please check your data and model files.")
