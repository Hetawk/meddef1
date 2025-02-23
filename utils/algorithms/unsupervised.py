<<<<<<< HEAD
# unsupervised.py

from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import FastICA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class UnsupervisedLearning:

    # Clustering
    def kmeans_clustering(self, X, n_clusters=3):
        model = KMeans(n_clusters=n_clusters)
        model.fit(X)
        return model

    def gmm_clustering(self, X, n_components=3):
        model = GaussianMixture(n_components=n_components)
        model.fit(X)
        return model

    def dbscan_clustering(self, X, eps=0.5, min_samples=5):
        model = DBSCAN(eps=eps, min_samples=min_samples)
        model.fit(X)
        return model

    # Dimensionality Reduction
    def pca(self, X, n_components=2):
        model = PCA(n_components=n_components)
        X_reduced = model.fit_transform(X)
        return model, X_reduced

    def tsne(self, X, n_components=2, perplexity=30, n_iter=1000):
        model = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter)
        X_reduced = model.fit_transform(X)
        return model, X_reduced

    def ica(self, X, n_components=2):
        model = FastICA(n_components=n_components)
        X_reduced = model.fit_transform(X)
        return model, X_reduced

    # Anomaly Detection
    def isolation_forest(self, X, n_estimators=100):
        model = IsolationForest(n_estimators=n_estimators)
        model.fit(X)
        return model

    def local_outlier_factor(self, X, n_neighbors=20):
        model = LocalOutlierFactor(n_neighbors=n_neighbors)
        model.fit(X)
        return model
=======
# unsupervised.py

from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import FastICA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class UnsupervisedLearning:

    # Clustering
    def kmeans_clustering(self, X, n_clusters=3):
        model = KMeans(n_clusters=n_clusters)
        model.fit(X)
        return model

    def gmm_clustering(self, X, n_components=3):
        model = GaussianMixture(n_components=n_components)
        model.fit(X)
        return model

    def dbscan_clustering(self, X, eps=0.5, min_samples=5):
        model = DBSCAN(eps=eps, min_samples=min_samples)
        model.fit(X)
        return model

    # Dimensionality Reduction
    def pca(self, X, n_components=2):
        model = PCA(n_components=n_components)
        X_reduced = model.fit_transform(X)
        return model, X_reduced

    def tsne(self, X, n_components=2, perplexity=30, n_iter=1000):
        model = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter)
        X_reduced = model.fit_transform(X)
        return model, X_reduced

    def ica(self, X, n_components=2):
        model = FastICA(n_components=n_components)
        X_reduced = model.fit_transform(X)
        return model, X_reduced

    # Anomaly Detection
    def isolation_forest(self, X, n_estimators=100):
        model = IsolationForest(n_estimators=n_estimators)
        model.fit(X)
        return model

    def local_outlier_factor(self, X, n_neighbors=20):
        model = LocalOutlierFactor(n_neighbors=n_neighbors)
        model.fit(X)
        return model
>>>>>>> 16c5cfd9eac902321ee831908acfc69f3a52f936
