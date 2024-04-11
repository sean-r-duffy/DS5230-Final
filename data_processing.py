from pyspark.sql import SparkSession
import pyspark.pandas as ps
import requests
import gzip
import io


def load_data(local_copy=True, local_dir='', file_names=None):
    """
    :param local_copy: bool: If False, data will be downloaded from online repository
    :param local_dir: str: Directory containing local files, used if local_copy == True
    :param file_names: dict: Filenames for books, authors, genres, interactions, and reviews if different from repo
    :return: pyspark.pandas Dataframe containing books info
    """

    books_url = 'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_books.json.gz'
    authors_url = 'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_book_authors.json.gz'
    genres_url = 'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_book_genres_initial.json.gz'
    interactions_url = 'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_interactions.csv'
    reviews_url = 'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_reviews_spoiler_raw.json.gz'

    def extract_file_name(url):
        return url.split('/')[-1]

    default_file_names = {'books': extract_file_name(books_url),
                          'authors': extract_file_name(authors_url),
                          'genres': extract_file_name(genres_url),
                          'interactions': extract_file_name(interactions_url),
                          'reviews': extract_file_name(reviews_url)}

    if file_names:
        for key in [x for x in default_file_names.keys()]:
            if key not in file_names:
                file_names[key] = default_file_names[key]
    else:
        file_names = default_file_names

    if local_copy:
        books_path = local_dir + file_names['books']
        authors_path = local_dir + file_names['authors']
        genres_path = local_dir + file_names['genres']
        interactions_path = local_dir + file_names['interactions']
        reviews_path = local_dir + file_names['reviews']
    else:
        books_path = io.BytesIO(requests.get(books_url).content)
        authors_path = io.BytesIO(requests.get(authors_url).content)
        genres_path = io.BytesIO(requests.get(genres_url).content)
        interactions_path = io.BytesIO(requests.get(interactions_url).content)
        reviews_path = io.BytesIO(requests.get(reviews_url).content)

    books_df = ps.read_json(gzip.open(books_path), lines=True, index_col='book_id')
    authors_df = ps.read_json(gzip.open(authors_path), lines=True, index_col='author_id')
    genres_df = ps.read_json(gzip.open(genres_path), lines=True, index_col='book_id')
    reviews_df = ps.read_json(gzip.open(reviews_path), lines=True, index_col='book_id')
    int_df = ps.read_csv(interactions_path)

    books_df = books_df.drop(
        columns=['series', 'asin', 'kindle_asin', 'similar_books', 'link', 'url', 'image_url',
                 'edition_information', 'title_without_series', 'popular_shelves', 'publisher'])

    def extract_authors(authors_dict):
        return [author['author_id'] for author in authors_dict]

    authors_column = books_df['authors'].apply(extract_authors)
    books_df['author_id'] = authors_column
    books_df = books_df.explode('author_id').set_index('book_id')
    books_df['author_id'] = books_df['author_id'].fillna(0).astype('int64')
    books_df = books_df.join(authors_df, how='inner', on='author_id', lsuffix='_book', rsuffix='_author')

    genres_df = genres_df['genres'].apply(ps.Series).join(genres_df)
    genres_df = genres_df.drop(columns=['genres']).fillna(0)
    genres_df = genres_df.set_index('book_id')
    books_df = books_df.join(genres_df, how='inner', on='book_id')

    return books_df

def load_data2():
    local_copy = True
    local_dir = 'data/'
    file_names = None

    """
    :param local_copy: bool: If False, data will be downloaded from online repository
    :param local_dir: str: Directory containing local files, used if local_copy == True
    :param file_names: dict: Filenames for books, authors, genres, interactions, and reviews if different from repo
    :return: pyspark.pandas Dataframe containing books info
    """

    books_url = 'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_books.json.gz'
    authors_url = 'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_book_authors.json.gz'
    genres_url = 'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_book_genres_initial.json.gz'
    interactions_url = 'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_interactions.csv'
    reviews_url = 'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_reviews_spoiler_raw.json.gz'

    def extract_file_name(url):
        return url.split('/')[-1]

    default_file_names = {'books': extract_file_name(books_url),
                          'authors': extract_file_name(authors_url),
                          'genres': extract_file_name(genres_url),
                          'interactions': extract_file_name(interactions_url),
                          'reviews': extract_file_name(reviews_url)}

    if file_names:
        for key in [x for x in default_file_names.keys()]:
            if key not in file_names:
                file_names[key] = default_file_names[key]
    else:
        file_names = default_file_names

    if local_copy:
        books_path = local_dir + file_names['books']
        authors_path = local_dir + file_names['authors']
        genres_path = local_dir + file_names['genres']
        interactions_path = local_dir + file_names['interactions']
        reviews_path = local_dir + file_names['reviews']
    else:
        books_path = io.BytesIO(requests.get(books_url).content)
        authors_path = io.BytesIO(requests.get(authors_url).content)
        genres_path = io.BytesIO(requests.get(genres_url).content)
        interactions_path = io.BytesIO(requests.get(interactions_url).content)
        reviews_path = io.BytesIO(requests.get(reviews_url).content)