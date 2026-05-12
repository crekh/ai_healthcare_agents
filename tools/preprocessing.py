from sklearn.preprocessing import StandardScaler

def preprocess_data(df):

    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler