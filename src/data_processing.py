import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import os

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_column='TransactionStartTime'):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.date_column] = pd.to_datetime(X[self.date_column], errors='coerce')
        X['transaction_hour'] = X[self.date_column].dt.hour
        X['transaction_day'] = X[self.date_column].dt.day
        X['transaction_month'] = X[self.date_column].dt.month
        X['transaction_year'] = X[self.date_column].dt.year
        return X

class CustomerAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, customer_id='CustomerId'):
        self.customer_id = customer_id

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg_df = X.groupby(self.customer_id).agg({
            'Value': ['sum', 'mean', 'std', 'count'],
            'Amount': ['sum', 'mean', 'std']
        })
        agg_df.columns = [
            'total_value', 'avg_value', 'std_value', 'transaction_count',
            'total_amount', 'avg_amount', 'std_amount'
        ]
        agg_df.reset_index(inplace=True)
        return agg_df

def build_pipeline(numeric_features, categorical_features):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    return ColumnTransformer([
        ('num', num_pipeline, numeric_features),
        ('cat', cat_pipeline, categorical_features)
    ])

def process_and_save_data(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)

    # Step 1: Add date-time features
    df = DateFeatureExtractor().fit_transform(df)

    # Step 2: Aggregate numeric features per customer
    agg_df = CustomerAggregator().fit_transform(df)

    # Step 3: Get latest transaction categorical + timestamp info
    categorical_features = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId',
                            'ProductCategory', 'ChannelId', 'PricingStrategy', 'FraudResult']
    latest_tx = df.sort_values(by='TransactionStartTime').groupby('CustomerId').last().reset_index()

    # Step 4: Merge aggregated data with latest context + TransactionStartTime
    merged_df = pd.merge(
        agg_df,
        latest_tx[['CustomerId'] + categorical_features + ['TransactionStartTime']],
        on='CustomerId',
        how='left'
    )

    # Step 5: Define numeric features
    numeric_features = ['total_value', 'avg_value', 'std_value', 'transaction_count',
                        'total_amount', 'avg_amount', 'std_amount']

    # Step 6: Build transformation pipeline
    pipeline = build_pipeline(numeric_features, categorical_features)

    # Step 7: Transform data
    transformed_data = pipeline.fit_transform(merged_df)

     # Step 8: Build DataFrame with headers
    cat_encoded_cols = pipeline.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features)
    final_columns = numeric_features + list(cat_encoded_cols)

    # Convert to dense array if it's sparse
    if hasattr(transformed_data, 'toarray'):
        transformed_data = transformed_data.toarray()

    df_final = pd.DataFrame(transformed_data, columns=final_columns)
    df_final.insert(0, 'CustomerId', merged_df['CustomerId'].values)
    df_final['TransactionStartTime'] = merged_df['TransactionStartTime']
    
    # Step 9: Save output
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_final.to_csv(output_csv_path, index=False)
    print(f"✅ Processed data saved to {output_csv_path}")
