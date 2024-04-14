import pandas as pd

int_df = pd.read_csv('data/interactions_top_books.csv')

int_df.pivot(index='user_id', columns='book_id', values='rating').fillna(0).to_csv('data/ratings.csv')
int_df.pivot(index='user_id', columns='book_id', values='is_read').fillna(0).to_csv('data/is_read.csv')
int_df.pivot(index='user_id', columns='book_id', values='shelved').fillna(0).to_csv('data/shelved.csv')
