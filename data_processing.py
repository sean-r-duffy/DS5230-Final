from pyspark.sql import SparkSession
import pyspark.pandas as ps

def load_data(local_copy=True, local_dir='data/'):
    books_url = 'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_books.json.gz'
    authors_url = 'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_book_authors.json.gz'
    genres_url = 'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_book_genres_initial.json.gz'
    interactions_url = 'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_interactions.csv'
    reviews_url = 'https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_reviews_spoiler_raw.json.gz'

    def extract_file_name(url):
        return url.split('/')[-1]
    
    books_path = local_dir + extract_file_name(books_url)
    authors_path = local_dir + extract_file_name(authors_url)
    genres_path = local_dir + extract_file_name(genres_url)
    interactions_path = local_dir + extract_file_name(interactions_url)
    reviews_path = local_dir + extract_file_name(reviews_url)

    if local_copy:
        books_df = ps.