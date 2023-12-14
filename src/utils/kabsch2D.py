import numpy as np
from numpy import ndarray


def centroid(X: ndarray) -> ndarray:
    """
    Centroid is the mean position of all the points in all of the coordinate
    directions, from a vectorset X.

    https://en.wikipedia.org/wiki/Centroid

    C = sum(X)/len(X)

    Parameters
    ----------
    X : array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    C : ndarray
        centroid
    """
    C: ndarray = X.mean(axis=0)
    return C

def kabsch2D(source: ndarray, target:ndarray, translate=False) -> ndarray:
    """
    Find optimal rotation matrix for sourcd toward target
    Three steps: 
    - find the centroids of both arrays 
    - compute their covariance matrix
    - then find optimal rotation U 

    Parameters
    ----------
    source : array
        (N, D) matrix, N is points and D is dimension

    target : array
        (N, D) matrix, N is points and D is dimension

    translate : Bool
        If True, compute translation term t (see https://nghiaho.com/?page_id=671)

    Returns : 
    rotated_source: array,
        (N, D) matrix, rotated sources based on target, using kabsch algorithm
        see more: http://en.wikipedia.org/wiki/Kabsch_algorithm

    t : array
        (N, D) translation matrix 

    """
    source_C = source - centroid(source) 
    target_C = target - centroid(target)
    
    # This should be D*D and not N*N ! 
    cov = np.dot(source_C.T, target_C)

    # Compute the optimal rotation matrix U using singluar value decomposition (SVD) 
    # This might need a correction depending on the sign, check later if we run into issues
    U, S, V = np.linalg.svd(cov)

    # Compute d to find its sign 
    d = np.linalg.det(np.dot(V, U.T)) < 0.0
    if d:
        U[:, -1] = -U[:, -1]


    # Wikipedia indicates a transpose here, but seems like it is not correct
    U = np.dot(U, V)

    if translate: 
        t = centroid(target) - centroid(source) @ U
        return U, t
    return U 
    

def rotate(source: ndarray, U: ndarray) -> ndarray:
    """
    rotate the source matrix using the U optimal rotation matrix found
    using the kabasch2D function

    Parameters
    -----------
    source : array
        (N, D) matrix, N is points and D is dimension

    U : array
        (D, D) optimal rotation matrix

    Returns
    --------
    source_rotated : array
        (N, D) matrix, N is points and D is dimension
    """
    return np.dot(source, U)

def error(rotated_source: ndarray, target: ndarray, t=None) -> float:
    """
    Find the error between a rotated source matrix (using the U optimal rotation found before)
    Computed by taking the sum of squared error
    """
    if t is not None: 
        diff = rotated_source + t - target
    else: 
        diff = rotated_source - target
    return (diff * diff).sum()


    
