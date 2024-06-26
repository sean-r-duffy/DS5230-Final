import pandas as pd
import gzip
import io
import requests

PANDAS = True
LOCAL_COPIES = True

BOOKS_URL = 'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_books.json.gz'
AUTHORS_URL = 'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_book_authors.json.gz'
GENRES_URL = 'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_book_genres_initial.json.gz'
INTERACTIONS_URL = 'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_interactions.csv'
REVIEWS_URL = 'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_reviews_spoiler_raw.json.gz'

BOOKS_PATH = 'data/goodreads_books.json.gz'
AUTHORS_PATH = 'data/goodreads_book_authors.json.gz'
GENRES_PATH = 'data/goodreads_book_genres_initial.json.gz'
INTERACTIONS_PATH = 'data/goodreads_interactions.csv'
REVIEWS_PATH = 'data/goodreads_reviews_spoiler_raw.json.gz'

CHUNKSIZE = None
KEEP_COLS = ['text_reviews_count', 'average_rating', 'num_pages', 'publication_year', 'book_id', 'ratings_count']

MIN_READ = 9986  # Gives top 1000 books

if LOCAL_COPIES:
    books_file = BOOKS_PATH
    genres_file = GENRES_PATH
    interactions_file = INTERACTIONS_PATH
else:
    books_file = BOOKS_URL
    genres_file = GENRES_URL
    interactions_file = INTERACTIONS_URL


def load_books():
    books_df = pd.read_json(books_file, lines=True, chunksize=CHUNKSIZE)
    genres_df = pd.read_json(genres_file, lines=True)

    if CHUNKSIZE:
        header = True
        for chunk in books_df:
            chunk = chunk[KEEP_COLS]
            chunk.to_csv('data/books.csv', mode='a', header=header)
            header = False
        books_df = pd.read_csv('data/books.csv')

    genres_df = genres_df[['genres']].apply(pd.Series).join(genres_df)
    genres_df = genres_df.drop(columns=['genres']).fillna(0)
    genres_df = genres_df.set_index('book_id')
    genres_df = genres_df.apply(lambda x: x / x.sum(), axis=1)

    books_df = books_df.merge(genres_df, on='book_id', how='right')
    books_df = books_df[(books_df.fiction >= 0) & (books_df.fiction <= 1)]
    books_df = books_df.fillna(books_df.median())
    books_df.to_csv('data/books.csv')


def load_interactions(ratings=True, read=True, shelved=True):
    int_df = pd.read_csv(interactions_file)
    read_counts = int_df.groupby(by='book_id').sum().sort_values('is_read', ascending=False)
    read_counts = read_counts[read_counts.is_read >= MIN_READ]
    int_df = int_df[int_df.book_id.isin(read_counts.index)]
    int_df['shelved'] = 1
    int_df.to_csv('data/interactions.csv', index=False)

    if ratings:
        ratings_df = int_df.pivot(index='user_id', columns='book_id', values='rating').fillna(0)
        ratings_df.to_csv('data/ratings.csv')
        del ratings_df

    if read:
        read_df = int_df.pivot(index='user_id', columns='book_id', values='is_read').fillna(0)
        read_df.to_csv('data/read.csv')
        del read_df

    if shelved:
        shelved_df = int_df.pivot(index='user_id', columns='book_id', values='is_read').fillna(0)
        shelved_df.to_csv('data/shelved.csv')
        del shelved_df




