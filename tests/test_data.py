import pandas as pd

def test_data_quality():
    df = pd.read_csv('../data/generated_data.csv')
    assert not df.duplicated().any(), "Duplicate records found"
    assert df['amount'].min() > 0, "Invalid transaction amount"
    assert pd.api.types.is_datetime64_any_dtype(pd.to_datetime(df['time'])), "Invalid time format"
    assert set(df['is_fraud']) == {0, 1}, "Invalid fraud labels"
    print("✅ Data validation passed!")

if __name__ == "__main__":
    test_data_quality()