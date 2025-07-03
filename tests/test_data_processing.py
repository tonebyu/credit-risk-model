import pandas as pd
import numpy as np
from src.data_processing import DateFeatureExtractor, CustomerAggregator

def test_date_feature_extractor():
    # Sample input data
    df = pd.DataFrame({
        'TransactionStartTime': ['2023-01-01 12:00:00', '2023-02-15 18:45:00']
    })

    transformer = DateFeatureExtractor()
    transformed_df = transformer.transform(df)

    # Check that new columns exist
    assert 'transaction_hour' in transformed_df.columns
    assert 'transaction_day' in transformed_df.columns
    assert 'transaction_month' in transformed_df.columns
    assert 'transaction_year' in transformed_df.columns

    # Check values
    assert transformed_df['transaction_hour'].iloc[0] == 12
    assert transformed_df['transaction_month'].iloc[1] == 2

def test_customer_aggregator():
    # Sample input data
    df = pd.DataFrame({
        'CustomerId': [1, 1, 2],
        'Value': [100, 150, 200],
        'Amount': [20, 30, 40]
    })

    aggregator = CustomerAggregator()
    result = aggregator.transform(df)

    # Check shape and content
    assert 'total_value' in result.columns
    assert 'avg_amount' in result.columns
    assert result[result['CustomerId'] == 1]['total_value'].values[0] == 250
    assert result[result['CustomerId'] == 2]['transaction_count'].values[0] == 1
