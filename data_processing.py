#import modin.pandas as pd
import pandas as pd
import gzip


def load_to_csv():
    books_path = 'data/goodreads_books.json.gz'
    authors_path = 'data/goodreads_book_authors.json.gz'
    genres_path = 'data/goodreads_book_genres_initial.json.gz'
    interactions_path = 'data/goodreads_interactions.csv'
    reviews_path = 'data/goodreads_reviews_spoiler_raw.json.gz'
    chunksize = 10000

    books_df = pd.read_json(gzip.open(books_path), lines=True, chunksize=chunksize)
    authors_df = pd.read_json(gzip.open(authors_path), lines=True, chunksize=chunksize)
    genres_df = pd.read_json(gzip.open(genres_path), lines=True, chunksize=chunksize)
    reviews_df = pd.read_json(gzip.open(reviews_path), lines=True, chunksize=chunksize)
    int_df = pd.read_csv(interactions_path, chunksize=chunksize)

    header = True
    for chunk in books_df:
        chunk = chunk[['isbn', 'text_reviews_count', 'country_code', 'language_code', 'average_rating', 'num_pages',
                       'isbn13', 'publication_year', 'book_id', 'ratings_count', 'title']]
        chunk.to_csv('data/books.csv', mode='a', header=header)
        header = False

    genres_df = pd.read_json(gzip.open(genres_path), lines=True)
    genres_df = genres_df['genres'].apply(pd.Series).join(genres_df)
    genres_df = genres_df.drop(columns=['genres']).fillna(0)
    genres_df = genres_df.set_index('book_id')
    genres_df = genres_df.apply(lambda x: x / x.sum(), axis=1)

    books_df = pd.read_csv('data/books.csv')
    books_df = books_df.merge(genres_df, on='book_id', how='right')
    books_df.to_csv('data/books_genres.csv')
