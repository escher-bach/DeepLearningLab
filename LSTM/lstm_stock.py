import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR =  "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


SEQ_LEN = 60
BATCH_SIZE = 32

def download_stock_data(ticker="AAPL", period="5y"):
    import yfinance as yf
    df = yf.download(ticker, period=period, auto_adjust=True)
    return df

class StockDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.seq_len], dtype=torch.float32)
        y = torch.tensor(self.data[idx + self.seq_len], dtype=torch.float32)
        return x, y


class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred.squeeze(), y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        loss = criterion(pred.squeeze(), y.squeeze())
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def main():
    print(f"Device: {DEVICE}")

    TICKER = "AAPL"
    print(f"Downloading {TICKER} stock data...")
    df = download_stock_data(TICKER, period="5y")
    prices = df["Close"].values.reshape(-1, 1).astype(np.float64)
    print(f"Total data points: {len(prices)}")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    split = int(len(scaled) * 0.8)
    train_data = scaled[:split]
    test_data = scaled[split:]

    train_ds = StockDataset(train_data, SEQ_LEN)
    test_ds = StockDataset(test_data, SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = StockLSTM(input_size=1, hidden_size=64, num_layers=2).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 50
    history = {"train_loss": [], "test_loss": []}

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_epoch(model, train_loader, criterion, optimizer)
        te_loss = evaluate(model, test_loader, criterion)
        history["train_loss"].append(tr_loss)
        history["test_loss"].append(te_loss)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{EPOCHS}  train_loss={tr_loss:.6f}  test_loss={te_loss:.6f}")


    # Loss curve
    plt.figure(figsize=(10, 4))
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["test_loss"], label="Test")
    plt.title("LSTM Stock Price Prediction  Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=150)
    plt.close()

    # Predict on full test set for visualization
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            pred = model(x).cpu().numpy()
            all_preds.append(pred)
            all_true.append(y.numpy())

    all_preds = np.concatenate(all_preds).reshape(-1, 1)
    all_true = np.concatenate(all_true).reshape(-1, 1)

    pred_prices = scaler.inverse_transform(all_preds)
    true_prices = scaler.inverse_transform(all_true)

    plt.figure(figsize=(14, 5))
    plt.plot(true_prices, label="Actual Price", linewidth=1.5)
    plt.plot(pred_prices, label="Predicted Price", linewidth=1.5, alpha=0.8)
    plt.title(f"{TICKER} Stock Price Actual vs Predicted")
    plt.xlabel("Time Step")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "prediction_vs_actual.png"), dpi=150)
    plt.close()

    rmse = np.sqrt(np.mean((pred_prices - true_prices) ** 2))
    mae = np.mean(np.abs(pred_prices - true_prices))

    with open(os.path.join(OUTPUT_DIR, "results.txt"), "w") as f:
        f.write(f"Ticker: {TICKER}\n")
        f.write(f"Sequence Length: {SEQ_LEN}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Train samples: {len(train_ds)}\n")
        f.write(f"Test samples: {len(test_ds)}\n")
        f.write(f"Final Train Loss (MSE): {history['train_loss'][-1]:.6f}\n")
        f.write(f"Final Test Loss (MSE): {history['test_loss'][-1]:.6f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "lstm_stock_model.pth"))
    print(f"\nOutputs saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
