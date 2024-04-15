import pandas as pd
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd


def perform_pca(dataframe, name, n_components=None):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(dataframe)
    pca = PCA(n_components=n_components, random_state=41)
    pca_components = pca.fit_transform(df_scaled)

    pca_df = pd.DataFrame(pca_components)
    pca_df.to_csv(f'{name}_pca_components.csv', index=False)

    info_df = pd.DataFrame({
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'singular_values': pca.singular_values_
    })
    info_df.to_csv(f'{name}_pca_info.csv', index=False)


def perform_svd(dataframe, name):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(dataframe)
    u, s, vt = svd(df_scaled)

    u_df = pd.DataFrame(u)
    u_df.to_csv(f'{name}_svd_U.csv', index=False)

    v_df = pd.DataFrame(vt.T)
    v_df.to_csv(f'{name}_svd_V.csv', index=False)

    s_df = pd.DataFrame(s)
    s_df.to_csv(f'{name}_svd_S.csv', index=False)


def perform_nmf(dataframe, name, n_components=None):
    nmf = NMF(n_components=n_components, init='random', random_state=41)
    nmf_components = nmf.fit_transform(dataframe)

    nmf_df = pd.DataFrame(nmf_components)
    nmf_df.to_csv(f'{name}_nmf_components.csv', index=False)

