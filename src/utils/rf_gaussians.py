import numpy as np

def gaussian_2d_curve(independent, x0, y0, sigma, slope, intercept):
    X, Y = independent
    x = X - x0
    y = Y - y0
    return (np.exp(-0.5 * ((x/sigma)**2 + (y/sigma)) * slope + intercept))
