import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def pca_scatter(X, y, title="PCA (2D)"):
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)
    plt.figure()
    for cls in np.unique(y):
        idx = (y == cls)
        plt.scatter(Z[idx, 0], Z[idx, 1], label=str(cls), s=12)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(title)
    plt.legend(markerscale=2)
    plt.tight_layout()

def tsne_scatter(X, y, title="t-SNE (2D)", perplexity=30, n_iter=1000):
    # For sklearn >= 1.4, use max_iter instead of n_iter
    try:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=n_iter,             # ✅ new name
            random_state=42,
            init="pca",
            learning_rate="auto"
        )
    except TypeError:
        # fallback for older sklearn versions
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=42,
            init="pca",
            learning_rate="auto"
        )

    Z = tsne.fit_transform(X)
    plt.figure()
    for cls in np.unique(y):
        idx = (y == cls)
        plt.scatter(Z[idx, 0], Z[idx, 1], label=str(cls), s=12)
    plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
    plt.title(title)
    plt.legend(markerscale=2)
    plt.tight_layout()
