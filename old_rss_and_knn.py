import matplotlib

matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1337)


############### GENERATING THE DATA ################
n_clusters = 10
points_per_class = 100

blue_means = np.random.multivariate_normal(
    mean=[1, 0],
    cov=np.eye(2),
    size=n_clusters,
)

orange_means = np.random.multivariate_normal(
    mean=[0, 1],
    cov=np.eye(2),
    size=n_clusters,
)

############### GENERATE OBSERVATIONS ##############


# low variance
cov = np.eye(2) / 5

blue = np.zeros((points_per_class, 2))
orange = np.zeros((points_per_class, 2))

for i in range(points_per_class):
    # randomly choose cluster mk
    k = np.random.randint(n_clusters)
    blue[i] = np.random.multivariate_normal(mean=blue_means[k], cov=cov)
    k = np.random.randint(n_clusters)
    orange[i] = np.random.multivariate_normal(mean=orange_means[k], cov=cov)

X = np.vstack((blue, orange))
y = np.concatenate((np.zeros(points_per_class), np.ones(points_per_class)))

plt.figure(figsize=(6, 6))

plt.scatter(blue[:, 0], blue[:, 1], color="steelblue", label="Blue")

plt.scatter(orange[:, 0], orange[:, 1], color="darkorange", label="Orange")

plt.legend()
plt.axis("equal")
# plt.show()

################# RSS (least-squares classifier) #########################
X_design = np.column_stack((np.ones(len(X)), X))

# Compute: ^B = (X^TX)^-1X^Ty
beta = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
pred = X_design @ beta

rss_class = (pred > 0.5).astype(int)


#################### kNN ########################


def knn_predict(trainX, trainY, x, k=15):
    d = np.sum((trainX - x) ** 2, axis=1)
    # skip first neighbour as that is the point itself (0 distance)
    nearest = np.argsort(d)[1:k+1]

    return np.round(np.mean(trainY[nearest]))


knn_class = np.zeros(len(X))
for i in range(len(X)):
    knn_class[i] = knn_predict(
        X,
        y,
        X[i],
        k=15,
    )

############### DECISION BOUNDARY ##################
x1 = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)

x2 = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)

xx, yy = np.meshgrid(x1, x2)

grid = np.column_stack((xx.ravel(), yy.ravel()))


#### RSS boundary ###
grid_design = np.column_stack((np.ones(len(grid)), grid))
rss_grid = (grid_design @ beta > 0.5).astype(int).reshape(xx.shape)

### kNN boundary ###

knn_grid = np.zeros(len(grid))
for i, p in enumerate(grid):
    knn_grid[i] = knn_predict(X, y, p, k=15)
knn_grid = knn_grid.reshape(xx.shape)


#### final plot

rss_pred = (X_design @ beta > 0.5).astype(int)
knn_pred = np.array([knn_predict(X, y, x, k=15) for x in X])

rss_error = np.mean(rss_pred != y)
knn_error = np.mean(knn_pred != y)

print(f"RSS error: {rss_error:.3f}")
print(f"kNN error: {knn_error:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].contourf(xx, yy, rss_grid, alpha=0.3, cmap="coolwarm")

ax[0].scatter(blue[:, 0], blue[:, 1], color="blue", s=15)

ax[0].scatter(orange[:, 0], orange[:, 1], color="orange", s=15)

ax[0].set_title(f"RSS\nError = {rss_error:.3f}")

ax[1].contourf(xx, yy, knn_grid, alpha=0.3, cmap="coolwarm")

ax[1].scatter(blue[:, 0], blue[:, 1], color="blue", s=15)

ax[1].scatter(orange[:, 0], orange[:, 1], color="orange", s=15)

ax[1].set_title(f"kNN (k=15)\nError = {knn_error:.3f}")

plt.show()
