import polars as pl
import numpy as np
from scipy.optimize import nnls
from sklearn.neighbors import NearestNeighbors


w = pl.read_csv('nmf/w.csv').to_numpy()
h = pl.read_csv('nmf/h.csv').to_numpy()
all_user_ids = pl.read_csv('user_subset.csv')
all_book_ids = pl.read_csv('books_subset.csv')
all_book_ids = [str(x[0]) for x in all_book_ids.to_numpy()]

interactions = pl.read_parquet('ratings_subset_large.parquet')


def get_user_ratings(user_ids):
    ratings = pl.DataFrame()
    for uid in user_ids:
        ratings = ratings.vstack(interactions.filter(pl.col('user_id')==uid))
    #ratings = interactions.filter(pl.col('user_id').is_in(user_ids))
    ratings = ratings.select(all_book_ids)
    ratings = ratings.fill_null(0).to_numpy()
    return ratings


def get_user_embeddings(user_ratings):
    embeddings = np.array([nnls(h.T, x)[0] for x in user_ratings])
    return embeddings


def get_user_predictions(user_embeddings):
    predicted_ratings = np.dot(user_embeddings, h)
    return predicted_ratings


def predict(user_ids):
    ratings = get_user_ratings(user_ids)
    embeddings = get_user_embeddings(ratings)
    predicted_ratings = get_user_predictions(embeddings)
    predicted_ratings = pl.DataFrame(predicted_ratings)
    predicted_ratings.columns = all_book_ids
    user_id_column = pl.DataFrame({'user_id': user_ids})
    predicted_ratings = pl.concat([user_id_column, predicted_ratings], how='horizontal')
    return predicted_ratings


def top_books(df, n=10):
    df = df.melt(id_vars='user_id', value_vars=all_book_ids, value_name='rating', variable_name='book_id')
    df = df.sort(['user_id','rating'], descending=[False, True])
    df = df.with_columns(pl.Series('ranking', [x for x in range(1, len(all_book_ids)+1)]*df.select(pl.col('user_id')).n_unique()))
    df = df.filter(pl.col('ranking')<=n)
    df = df.with_columns([pl.col('book_id').cast(pl.Int32),
                          pl.col('ranking').cast(pl.Int8),
                          pl.col('user_id').cast(pl.Int32)])
    return df


def predict_top_books(user_ids, n=10):
    predictions = predict(user_ids)
    predictions = top_books(predictions, n=n)
    return predictions


def get_top_books(user_ids, n=10, subset=True):
    ratings = interactions.filter(pl.col('user_id').is_in(user_ids))
    if subset:
        ratings = ratings.select(['user_id']+all_book_ids)
    ratings = ratings.fill_null(0)
    top = top_books(ratings, n=n)
    return top


def get_book_ratings(book_ids):
    ratings = interactions.select(['user_id']+book_ids)
    ratings = ratings.filter(pl.col('user_id').is_in(all_user_ids))
    ratings = ratings.drop('user_id')
    ratings = ratings.fill_null(0).to_numpy().T
    return ratings


def get_book_embeddings(book_ratings):
    embeddings = np.array([nnls(w, x)[0] for x in book_ratings])
    return embeddings


def get_nearest_books(book_ids, n=10):
    nn = NearestNeighbors(n_neighbors=n, metric='cosine')
    ratings = get_book_ratings(book_ids)
    embeddings = get_book_embeddings(ratings)
    nn.fit(h.T)
    distances, indices = nn.kneighbors(embeddings)
    look_up = np.array(all_book_ids)
    neighbors = np.array([look_up[x] for x in indices]).squeeze()
    neighbors = pl.concat([pl.DataFrame({'book_id': book_ids}), pl.DataFrame(neighbors)], how='horizontal')
    neighbors.columns = ['book_id']+[str(x) for x in range(1, n+1)]
    neighbors = neighbors.with_columns([pl.col(x).cast(pl.Int32) for x in neighbors.columns])
    return neighbors


def get_nearest_users(user_ids, n=10):
    nn = NearestNeighbors(n_neighbors=n, metric='cosine')
    ratings = get_user_ratings(user_ids)
    embeddings = get_user_embeddings(ratings)
    nn.fit(w)
    distances, indices = nn.kneighbors(embeddings)
    look_up = all_user_ids.to_numpy()
    neighbors = np.array([look_up[x] for x in indices]).squeeze()
    neighbors = pl.concat([pl.DataFrame({'user_id': user_ids}), pl.DataFrame(neighbors)], how='horizontal')
    neighbors.columns = ['user_id']+[str(x) for x in range(1, n+1)]
    return neighbors

