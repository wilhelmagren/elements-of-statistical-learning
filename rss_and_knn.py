import numpy as np
import matplotlib.pyplot as plt


def generate_centers(n_clusters=10):
    """
    Generate the Gaussian mixture centers.

    Returns
    -------
    blue_centers : (n_clusters, 2) ndarray
    orange_centers : (n_clusters, 2) ndarray
    """

    blue_centers = np.random.multivariate_normal(
        mean=[1, 0], cov=np.eye(2), size=n_clusters
    )

    orange_centers = np.random.multivariate_normal(
        mean=[0, 1], cov=np.eye(2), size=n_clusters
    )

    return blue_centers, orange_centers


def sample_dataset(
    blue_centers, orange_centers, points_per_class=100, cluster_cov=None
):
    """
    Sample one dataset from the fixed Gaussian mixture.

    Parameters
    ----------
    blue_centers : ndarray
        Centers for the blue class.

    orange_centers : ndarray
        Centers for the orange class.

    points_per_class : int
        Number of observations per class.

    cluster_cov : ndarray
        Covariance matrix of each small Gaussian cluster.

    Returns
    -------
    X : (2*points_per_class, 2) ndarray
    y : (2*points_per_class,) ndarray
    """

    if cluster_cov is None:
        cluster_cov = np.eye(2) / 5

    n_clusters = len(blue_centers)

    blue = np.zeros((points_per_class, 2))
    orange = np.zeros((points_per_class, 2))

    # Generate blue observations
    for i in range(points_per_class):
        k = np.random.randint(n_clusters)
        blue[i] = np.random.multivariate_normal(blue_centers[k], cluster_cov)

    # Generate orange observations
    for i in range(points_per_class):
        k = np.random.randint(n_clusters)
        orange[i] = np.random.multivariate_normal(orange_centers[k], cluster_cov)

    X = np.vstack((blue, orange))

    y = np.concatenate(
        [np.zeros(points_per_class, dtype=int), np.ones(points_per_class, dtype=int)]
    )

    return X, y


np.random.seed(0)

# Generate the underlying distribution
blue_centers, orange_centers = generate_centers()

# Independent samples from the same distribution
X_train, y_train = sample_dataset(blue_centers, orange_centers)

X_test, y_test = sample_dataset(blue_centers, orange_centers)

import numpy as np


class RSSClassifier:
    """
    Least squares classifier.

    Fits:
        y = beta_0 + beta_1*x1 + beta_2*x2

    Prediction:
        y_hat > 0.5 -> class 1
        y_hat <= 0.5 -> class 0
    """

    def __init__(self):
        self.beta = None

    def fit(self, X, y):
        """
        Fit least squares coefficients.
        """

        # Add intercept column
        X_design = np.column_stack((np.ones(len(X)), X))

        # Solve least squares
        self.beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)

        return self

    def predict_proba(self, X):
        """
        Return continuous predictions.
        """

        X_design = np.column_stack((np.ones(len(X)), X))

        return X_design @ self.beta

    def predict(self, X):
        """
        Return class predictions.
        """

        return (self.predict_proba(X) > 0.5).astype(int)


class KNNClassifier:
    """
    k-nearest neighbors classifier.
    """

    def __init__(self, k=15):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Store training data.
        """

        self.X_train = X
        self.y_train = y

        return self

    def predict_one(self, x):
        """
        Predict one observation.
        """

        # Squared Euclidean distance
        distances = np.sum((self.X_train - x) ** 2, axis=1)

        # Find k closest points
        nearest = np.argsort(distances)[: self.k]

        # Majority vote
        return int(np.mean(self.y_train[nearest]) > 0.5)

    def predict(self, X):
        """
        Predict many observations.
        """

        predictions = []

        for x in X:
            predictions.append(self.predict_one(x))

        return np.array(predictions)


# ============================
# Fit models
# ============================

rss = RSSClassifier()

rss.fit(X_train, y_train)


knn = KNNClassifier(k=15)

knn.fit(X_train, y_train)


# ============================
# Predictions
# ============================

rss_train_pred = rss.predict(X_train)
rss_test_pred = rss.predict(X_test)

knn_train_pred = knn.predict(X_train)
knn_test_pred = knn.predict(X_test)


# ============================
# Errors
# ============================

rss_train_error = np.mean(rss_train_pred != y_train)

rss_test_error = np.mean(rss_test_pred != y_test)


knn_train_error = np.mean(knn_train_pred != y_train)

knn_test_error = np.mean(knn_test_pred != y_test)


print("RSS")
print(f"Train error: {rss_train_error:.3f}")
print(f"Test error : {rss_test_error:.3f}")

print()

print("kNN")
print(f"Train error: {knn_train_error:.3f}")
print(f"Test error : {knn_test_error:.3f}")


# ============================
# Create prediction grid
# ============================


def create_grid(X, resolution=300):

    x_min = X[:, 0].min() - 1
    x_max = X[:, 0].max() + 1

    y_min = X[:, 1].min() - 1
    y_max = X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
    )

    grid = np.column_stack((xx.ravel(), yy.ravel()))

    return xx, yy, grid


# Use both train and test points to define plot range
X_all = np.vstack((X_train, X_test))

xx, yy, grid = create_grid(X_all)


# ============================
# Grid predictions
# ============================

rss_grid = rss.predict(grid)

knn_grid = knn.predict(grid)


rss_grid = rss_grid.reshape(xx.shape)

knn_grid = knn_grid.reshape(xx.shape)


# ============================
# Plot helper
# ============================


def plot_result(ax, xx, yy, boundary, X, y, title):

    ax.contourf(xx, yy, boundary, alpha=0.25, cmap="coolwarm")

    ax.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        color="steelblue",
        edgecolor="black",
        s=25,
        label="Blue",
    )

    ax.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        color="darkorange",
        edgecolor="black",
        s=25,
        label="Orange",
    )

    ax.set_title(title)
    ax.set_aspect("equal")


# ============================
# Final 2x2 plot
# ============================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))


plot_result(
    axes[0, 0],
    xx,
    yy,
    rss_grid,
    X_train,
    y_train,
    f"RSS - Training\nError = {rss_train_error:.3f}",
)


plot_result(
    axes[0, 1],
    xx,
    yy,
    rss_grid,
    X_test,
    y_test,
    f"RSS - Test\nError = {rss_test_error:.3f}",
)


plot_result(
    axes[1, 0],
    xx,
    yy,
    knn_grid,
    X_train,
    y_train,
    f"kNN (k=15) - Training\nError = {knn_train_error:.3f}",
)


plot_result(
    axes[1, 1],
    xx,
    yy,
    knn_grid,
    X_test,
    y_test,
    f"kNN (k=15) - Test\nError = {knn_test_error:.3f}",
)


plt.tight_layout()
plt.show()
