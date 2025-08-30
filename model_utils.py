import os
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, Model


# -----------------------
# Sliding window
# -----------------------
def create_sequences(X, y, time_steps=20):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


# -----------------------
# Positional Encoding
# -----------------------
def positional_encoding(maxlen, d_model):
    pos = np.arange(maxlen)[:, None]
    i = np.arange(d_model)[None, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angles = pos * angle_rates
    pe = np.zeros((maxlen, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.constant(pe, dtype=tf.float32)


class AddPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, maxlen, d_model, **kwargs):
        # Pass extra kwargs to Layer, so Keras can handle trainable/dtype/etc
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.d_model = d_model
        # create positional encoding matrix
        pos = np.arange(maxlen)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates
        # apply sin to even indices in the array; cos to odd indices
        pos_encoding = np.zeros((maxlen, d_model))
        pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
        self.pos_encoding = tf.constant(pos_encoding, dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding



# -----------------------
# Transformer Encoder Block
# -----------------------
def encoder_block(x, n_heads, dff, dropout):
    attn_out = layers.MultiHeadAttention(num_heads=n_heads, key_dim=x.shape[-1])(x, x)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    ffn = layers.Dense(dff, activation="relu")(x)
    ffn = layers.Dropout(dropout)(ffn)
    ffn = layers.Dense(x.shape[-1])(ffn)
    x = layers.Add()([x, ffn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x


# -----------------------
# Baseline Forecast (Naive)
# -----------------------
def baseline_forecast(df, target_col):
    y_true = df[target_col].values[1:]
    preds = df[target_col].shift(1).dropna().values
    return preds, y_true


# -----------------------
# Train + Save Transformer
# -----------------------
def train_transformer_model(
    df, target_col, save_dir=".", time_steps=20,
    d_model=64, n_heads=4, dff=128, n_layers=2,
    dropout=0.2, epochs=20, batch_size=32
):
    # Drop target column and Holiday_ID if present
    drop_cols = [target_col]
    if "Holiday_ID" in df.columns:
        drop_cols.append("Holiday_ID")

    X = df.drop(columns=drop_cols)
    y = df[[target_col]]

    # Drop datetime columns
    datetime_cols = X.select_dtypes(include=["datetime64[ns]"]).columns
    if len(datetime_cols) > 0:
        print(f"Dropping datetime columns: {list(datetime_cols)}")
        X = X.drop(columns=datetime_cols)

    # Scale features
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X.values)
    y_scaled = scaler_y.fit_transform(y.values)

    # Train-validation split
    split = int(len(df) * 0.8)
    X_train, X_val = X_scaled[:split], X_scaled[split:]
    y_train, y_val = y_scaled[:split], y_scaled[split:]

    # Create sequences for Transformer
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, time_steps)

    n_features = X_train_seq.shape[2]

    # Build Transformer model
    inp = layers.Input(shape=(time_steps, n_features))
    x = layers.Dense(d_model)(inp)
    x = AddPositionalEncoding(time_steps, d_model)(x)

    for _ in range(n_layers):
        x = encoder_block(x, n_heads=n_heads, dff=dff, dropout=dropout)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1)(x)

    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")

    # Train
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Save model and scalers
    model_path = os.path.join(save_dir, "tsfm_model.keras")
    scaler_X_path = os.path.join(save_dir, "scaler_X.pkl")
    scaler_y_path = os.path.join(save_dir, "scaler_y.pkl")

    model.save(model_path)
    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)

    # Evaluate
    y_pred_scaled = model.predict(X_val_seq)
    y_val_inv = scaler_y.inverse_transform(y_val_seq.reshape(-1, 1))
    y_pred_inv = scaler_y.inverse_transform(y_pred_scaled)

    results = {
        "RMSE": float(np.sqrt(mean_squared_error(y_val_inv, y_pred_inv))),
        "MAE": float(mean_absolute_error(y_val_inv, y_pred_inv)),
        "MAPE": float(np.mean(np.abs((y_val_inv - y_pred_inv) / y_val_inv)) * 100),
        "R²": float(r2_score(y_val_inv, y_pred_inv))
    }

    return model, scaler_X, scaler_y, history, results



# -----------------------
# Forecast with Transformer
# -----------------------
def transformer_forecast(model, scaler_X, scaler_y, df, target_col, time_steps=20):
    X = df.drop(columns=[target_col])
    y = df[[target_col]]

    datetime_cols = X.select_dtypes(include=["datetime64[ns]"]).columns
    if len(datetime_cols) > 0:
        X = X.drop(columns=datetime_cols)

    X_scaled = scaler_X.transform(X.values)
    y_scaled = scaler_y.transform(y.values)

    X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

    y_pred_scaled = model.predict(X_seq)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_seq.reshape(-1, 1))

    return y_pred.flatten(), y_true.flatten()


# -----------------------
# Evaluation Utility
# -----------------------
def evaluate_model(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "R²": float(r2_score(y_true, y_pred))
    }
