import polars as pl
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from scipy import sparse

df = pl.read_parquet('ratings_subset_large.parquet')
book_ids = pl.DataFrame({'book_id': df.columns[1:]}).with_columns(pl.col('book_id').cast(pl.Int32))
user_ids = df.select('user_id')

def run_nmf(interactions_df, books, users):
    print('NMF Started')
    for n_comp  in [50, 100, 150, 200, 250]:
        try:
            label = f'{n_comp}_components_full'
            df = interactions_df
            df = df.drop('user_id')
            df = df.fill_null(0)
            df = sparse.csc_matrix(df)
            model = NMF(n_components=n_comp, init='random', solver='mu', max_iter=200, random_state=41)
            W = pl.DataFrame(model.fit_transform(df))
            H = pl.DataFrame(model.components_)
            V = pl.DataFrame(np.dot(W,H))
            W.write_csv(f'0421/w_{label}.csv')
            H.write_csv(f'0421/h_{label}.csv')
            V.write_csv(f'0421/v_{label}.csv')
            print(label+'_finished')
        except:
            continue
                
    for n_comp  in [50, 100, 150, 200, 250]:
        for users_books in [(5000, 100), (100000, 5000), (300000, 50000)]:
            try:
                label = f'{n_comp}_components_{users_books[0]}_users_{users_books[1]}_books'
                user_subset = users.sample(n=users_books[0], seed=41)
                book_subset = books.sample(n=users_books[1], seed=41)

                df = interactions_df
                df = df.filter(pl.col('user_id').is_in(user_subset.select(pl.col('user_id'))))
                df = df.select([str(x) for x in book_subset.get_column('book_id').to_list()])
                df = df.drop('user_id')
                df = df.fill_null(0)
                df = sparse.csc_matrix(df)
                model = NMF(n_components=n_comp, init='random', solver='mu', max_iter=200, random_state=41)
                W = pl.DataFrame(model.fit_transform(df))
                H = pl.DataFrame(model.components_)
                V = pl.DataFrame(np.dot(W,H))
                W.write_csv(f'0421/w_{label}.csv')
                H.write_csv(f'0421/h_{label}.csv')
                V.write_csv(f'0421/v_{label}.csv')
                print(label+'_finished')
            except:
                continue
                
run_nmf(df, book_ids, user_ids)

