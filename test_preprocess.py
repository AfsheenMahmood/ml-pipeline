from preprocess import load_data, preprocess_data

def test_no_missing_values():
    df = load_data()
    df = preprocess_data(df)
    assert df.isnull().sum().sum() == 0

def test_data_is_scaled():
    df = load_data()
    df = preprocess_data(df)
    max_val = df[['Age', 'Fare']].max().max()
    assert max_val < 3  # StandardScaler usually gives values between -3 and 3
