import pandas as pd

def load_data():
    df = pd.read_excel(r"C:\uni works\projects\SmartGridGuard\energy_forecasting_testcode\elf_dataset.xlsx")
    return df
