import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from pyclustertend import hopkins
from data1 import load_data, preprocessing

class KPrototype:
    def __init__(self, n_clusters=8, max_iter=300, n_inits=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_inits = n_inits

    def fit(self, X1, X2, NAcols):
        # X1: numerical values, X2: nominal values
        #print(X1, X2)
        self.best_inertia_ = 1e30
        for init in range(self.n_inits):
            idx = np.random.permutation(X1.shape[0])[:self.n_clusters]
            #center1 = np.zeros((self.n_clusters, X1.shape[1]), dtype=float)
            #center2 = np.zeros((self.n_clusters, X2.shape[1]), dtype=int)
            #assignments = np.random.randint(0, self.n_clusters, size=X1.shape[0])
            center1, center2 = X1[idx], X2[idx]
            assignments = np.zeros(X1.shape[0], dtype=int)
            cur_assignments = np.zeros(X1.shape[0], dtype=int)
            for i in range(self.max_iter):
                print("iter ", i)
                #print(center1, center2, assignments)
                for j in tqdm(range(X1.shape[0])):
                    dist = 1e30
                    for k in range(self.n_clusters):
                        cur = self.distance(X1[j], center1[k], X2[j], center2[k], NAcols)
                        #print(j, cur)
                        if cur < dist:
                            dist = cur
                            cur_assignments[j] = k
                #print(i, center, cluster, cur_cluster)
                if np.array_equal(assignments, cur_assignments):
                    print("Center points converged, early stop.")
                    break
                for i in range(X1.shape[0]):
                    assignments[i] = cur_assignments[i]
                for j in range(self.n_clusters):
                    if len(np.where(assignments == j)[0]) == 0:
                        idx = np.random.randint(0, X1.shape[0])
                        center1[j] = X1[idx]
                        center2[j] = X2[idx]
                    else:
                        center1[j], center2[j] = self.get_center(X1[np.where(assignments == j)], X2[np.where(assignments == j)], NAcols)
                cur_inertia = 0
                for i in range(X1.shape[0]):
                    cur_inertia += self.distance(X1[i], center1[assignments[i]], X2[i], center2[assignments[i]], NAcols)
                print("inertia=", cur_inertia)
            if cur_inertia < self.best_inertia_:
                self.best_inertia_ = cur_inertia
                self.labels_ = assignments
                self.cluster_centers_ = (center1, center2)
        
    def distance(self, x1, y1, x2, y2, NAcols):
        dist1 = np.sum(np.power(x1 - y1, 2))
        dist2 = np.sum(np.logical_or(x2 != y2, np.logical_or(x2 == NAcols, y2 == NAcols)))
        #print("getdistance", x1, y1, x2, y2, dist1, dist2)
        return dist1 + dist2

    def get_center(self, points1, points2, NAcols):
        c1 = np.mean(points1, axis=0)
        c2 = np.zeros(points2.shape[1], dtype=int)
        for i in range(points2.shape[1]):
            #print(np.bincount(points2[:, i]), np.argmax(np.bincount(points2[:, i])))
            x = np.bincount(points2[:, i])
            if NAcols[i] != -1:
                x[NAcols[i]] = 0
            c2[i] = np.argmax(x)
        #print("getcenter", points1, points2, c1, c2)
        return c1, c2

    def compactness(self, x1, x2, center1, center2, NAcols, assignments):
        ans = 0
        for i in range(len(center1)):
            cur = 0
            idxs = np.where(assignments == i)[0]
            for idx in idxs:
                cur = cur + self.distance(x1[idx], center1[i], x2[idx], center2[i], NAcols)
            ans += cur / len(idxs)
        return ans / len(center1) 

    def seperation(self, center1, center2, NAcols):
        ans, cnt = 0, 0
        for i in range(len(center1)):
            for j in range(i + 1, len(center1)):
                cnt += 1
                ans += self.distance(center1[i], center1[j], center2[i], center2[j], NAcols)
        return ans / cnt

if __name__ == "__main__":
    """
    np.random.seed(42)
    N = 50
    X1 = np.random.randn(N, 5)
    X2 = np.zeros((N, 1), dtype=int)
    model = KPrototype(n_clusters=3, n_inits=1)
    model.fit(X1, X2)
    print(model.labels_, model.cluster_centers_)
    center1, center2 = model.cluster_centers_
    print("Compactness=", model.compactness(X1, X2, center1, center2, model.labels_))
    print("Seperation=", model.seperation(center1, center2))

    """
    data = load_data('./diabetic_data.csv')
    print(data.shape)
    X1, X2, NAcols = preprocessing(data)
    print(X1.shape, X2.shape, NAcols)
    compactness = []
    seperation = []
    for i in range(1):
        model = KPrototype(n_clusters=10, n_inits=1, max_iter=5)
        model.fit(X1, X2, NAcols)
        #print("Hopkins statistic=", hopkins(np.concatenate((X1, X2), axis=1), X1.shape[0]))
        center1, center2 = model.cluster_centers_
        print(center1, center2)
        print("Compactness=", model.compactness(X1, X2, center1, center2, NAcols, model.labels_))
        print("Seperation=", model.seperation(center1, center2, NAcols))
        compactness.append(model.compactness(X1, X2, center1, center2, NAcols, model.labels_))
        seperation.append(model.seperation(center1, center2, NAcols))
    print(np.mean(compactness), '±', np.std(compactness))
    print(np.mean(seperation), '±', np.std(seperation))
    np.save("./cluster10.npy", model.labels_)
