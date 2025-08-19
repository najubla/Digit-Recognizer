import numpy as np
import pandas as pd

def load_kaggle_data(train_path="data/train.csv",
                     test_path="data/test.csv",
                     val_ratio=0.1,
                     seed=42):
    # 1) Read CSV
    df_train = pd.read_csv(train_path)
    y = df_train.iloc[:, 0].to_numpy(np.int64)          # (N,)
    X = df_train.iloc[:, 1:].to_numpy(np.float32)       # (N, 784)

    df_test = pd.read_csv(test_path)
    X_test = df_test.to_numpy(np.float32)               # (N_test, 784)

    # 2) Normalize and reshape (N,1,28,28)
    X = (X / 255.0).astype(np.float32).reshape(-1, 1, 28, 28)
    X_test = (X_test / 255.0).astype(np.float32).reshape(-1, 1, 28, 28)

    # 3) One-hot
    def _one_hot(y, num_classes=10):
        N = y.shape[0]
        out = np.zeros((N, num_classes), dtype=np.float32)
        out[np.arange(N), y] = 1.0
        return out

    y_oh = _one_hot(y, 10)  # (N, 10)

    # 4) Split train/val 
    def _split_train_val(X, y, val_ratio=0.1, seed=42):
        N = X.shape[0]
        rng = np.random.default_rng(seed)
        idx = np.arange(N)
        rng.shuffle(idx)
        n_val = int(N * val_ratio)
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]
        return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]

    X_tr, y_tr, X_val, y_val = _split_train_val(X, y_oh, val_ratio, seed)

    # 5) 
    return {
        "train": (X_tr, y_tr),
        "val":   (X_val, y_val),
        "test":  X_test
    }

if __name__ == "__main__":
    data = load_kaggle_data()
    (X_tr, y_tr), (X_val, y_val), X_test = data["train"], data["val"], data["test"]
    print(X_tr.shape, y_tr.shape)
    print(X_val.shape, y_val.shape)
    print(X_test.shape)
    assert X_tr.dtype == np.float32 and X_test.dtype == np.float32
    assert np.allclose(y_tr.sum(axis=1), 1.0)
