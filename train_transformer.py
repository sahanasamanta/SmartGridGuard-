if __name__ == "__main__":
    import pandas as pd
    from model_utils import train_transformer_model

    # Load dataset
    df = pd.read_excel("elf_dataset.xlsx")
    target_col = "DEMAND"

    # Train + Save
    model, scaler_X, scaler_y, history, results = train_transformer_model(df, target_col, save_dir=".")
    print("✅ Transformer model trained and saved successfully!")
    print("📊 Evaluation Results:", results)
