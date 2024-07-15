import numpy as np

def gaussian_2d_curve(independent, x0, y0, sigma, slope, intercept):
    X, Y = independent
    x = X - x0
    y = Y - y0
    return (np.exp(-0.5 * ((x/sigma)**2 + (y/sigma)**2)) * slope + intercept)

def gaussian_2d_curve_pol(independent, ecc, pol, sigma, slope, intercept):  
    X,Y=independent
    x0,y0 = pol2cart(ecc,pol)
    x = X - x0
    y = Y - y0
    return (np.exp(-0.5 * ((x/sigma)**2 + (y/sigma)**2)) * slope + intercept)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)