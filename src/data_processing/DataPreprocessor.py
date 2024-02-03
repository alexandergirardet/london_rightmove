




class DataPreprocessor:
    def __init__(self):
        pass
    @staticmethod
    def convert_frequencies(x):
        frequency = x['frequency']
        price = x['amount']

        if frequency == 'monthly':
            return price * 12
        elif frequency == 'weekly':
            return (price / 7) * 365
        elif frequency == 'daily':
            return price * 365
        elif frequency == 'quarterly':
            return price * 4
        else:  # Yearly
            return price

    @staticmethod
    def remove_anomalies(df, percentile_threshold=0.99):
        percentile_thresholds = df[['price', 'bedrooms', 'bathrooms']].quantile(percentile_threshold)

        # Filter the dataset to remove anomalies above the 98th percentile
        filtered_df = df[
            (df['price'] <= percentile_thresholds['price']) &
            (df['bedrooms'] <= percentile_thresholds['bedrooms']) &
            (df['bathrooms'] <= percentile_thresholds['bathrooms'])
            ]
        return filtered_df

    @staticmethod
    def merge_text(x):
        summary, feature_list = x[0], x[1]
        feature_list_joined = ', '.join(feature_list) if feature_list else ''
        return feature_list_joined + ' , ' + summary

    def preprocess_properties_with_binary(self, df):
        df['longitude'] = df['location'].apply(lambda x: x['longitude'])
        df['latitude'] = df['location'].apply(lambda x: x['latitude'])
        df = df.drop(columns=['location'])
        df['price'] = df['price'].apply(self.convert_frequencies)
        df['commercial'] = df['commercial'].apply(lambda x: 1 if x else 0)
        df['development'] = df['development'].apply(lambda x: 1 if x else 0)
        df['students'] = df['students'].apply(lambda x: 1 if x else 0)
        df['text'] = df[['summary', 'feature_list']].apply(self.merge_text, axis=1)
        df = self.remove_anomalies(df)
        return df

    def preprocess_properties(self, df):
        df['longitude'] = df['location'].apply(lambda x: x['longitude'])
        df['latitude'] = df['location'].apply(lambda x: x['latitude'])
        df = df.drop(columns=['location'])
        df['price'] = df['price'].apply(self.convert_frequencies)
        df['commercial'] = df['commercial'].apply(lambda x: 1 if x else 0)
        df['development'] = df['development'].apply(lambda x: 1 if x else 0)
        df['students'] = df['students'].apply(lambda x: 1 if x else 0)
        df['text'] = df[['summary', 'feature_list']].apply(self.merge_text, axis=1)
        df = self.remove_anomalies(df)
        return df

    @staticmethod
    def preprocess_walk_score(df):
        df = df.drop_duplicates(subset=['id'])
        df['walk_score'] = df['scores'].apply(lambda x: x['walk_score'])
        return df
