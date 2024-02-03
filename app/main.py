from data_processing.DataPreprocessor import DataPreprocessor

preprocessor = DataPreprocessor()

preprocessor.remove_anomalies(df, percentile_threshold=0.99)
