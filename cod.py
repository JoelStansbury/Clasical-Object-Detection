import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

def color_match(im, Q = 5, verbose = False):
    GMM_FEATURE_MATRIX = im.reshape(-1,3)
    model = KMeans(n_clusters=Q)
    CLOSEST_PRIMARY_COLORS = model.fit_predict(GMM_FEATURE_MATRIX)
    if verbose:
        c = model.cluster_centers_[CLOSEST_PRIMARY_COLORS]
        c = c.reshape(im.shape)
        plt.imshow(c.astype(int))
        plt.xticks([])
        plt.yticks([])
        plt.title('Primary colors found with KMeans')
        plt.show()
    return CLOSEST_PRIMARY_COLORS

def spacial_cluster(q, EPS = 5, verbose = False):
    model_2 = DBSCAN(eps=EPS)
    r = model_2.fit_predict(q)

    objects = []
    for j in range(r.max()+1):
        obj = q[np.where(r==j)]
        (x0,y0),(x1,y1) = obj.min(0),obj.max(0)
        cx,cy = ( (x0+x1)//2 ,(y0+y1)//2 )
        w,h = ( x1-x0, y1-y0 )
        objects.append([cx,cy,w,h])

        if verbose:
            plt.scatter(obj[:,0],obj[:,1],marker='.')
    if verbose:
        plt.show()
    return objects

def cod(im, Q=5, eps=5, verbose = False):
    CLOSEST_PRIMARY_COLORS = color_match(im, Q, verbose)
    compressed_image = CLOSEST_PRIMARY_COLORS.reshape(im.shape[:2])

    OBJECT_BBOXES = []
    for i in range(Q):
        q = np.flip(np.array(np.where(compressed_image==i)).T,1)
        OBJECT_BBOXES = OBJECT_BBOXES + spacial_cluster(q, eps, verbose)
    return OBJECT_BBOXES
