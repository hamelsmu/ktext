import pandas as pd
from ktext.preprocess import processor

data_url = 'https://storage.googleapis.com/issue_label_bot/pre_processed_data/processed_part0000.csv'
body = pd.read_csv(data_url).head(2000).text.tolist()
issue_body_proc = processor(heuristic_pct_padding=.7, keep_n=5000)
train_result = issue_body_proc.fit_transform(body)