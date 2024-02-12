from app.backend.data_processing import DataPreprocessor

preprocessor = DataPreprocessor()

preprocessor.remove_anomalies(df, percentile_threshold=0.99)
