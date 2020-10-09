import numpy as np

def MSE(X,Y, poly):
    """
    Calculate the Mean Squared Error (MSE) poly(X) and Y
    """
    return np.sum(( Y - poly(X) ) ** 2 ) / len(X)

def MSEderiv(X, Y, poly, n):
    """
    Calculate the partial derivative of the MSE on parameter order n
    """
    return np.sum( ( poly(X) - Y ) * np.power(X,n) ) / len(X)

def poly_reg(X, Y, n, lr, epochs):
    """
    Regression Function - Preform Polynomial regression with order n polynomial, return the best polynomial
    """

    theta = np.random.rand(n+1)

    for i in range(epochs):
        poly = np.poly1d(theta)

        #Evaluate Total Loss
        cost = MSE(X,Y, poly)

        print(cost)

        gradients = []
        #For each theta, calculate gradient
        for x in range(0,n+1):
            gradients.append( MSEderiv(X,Y,poly,x) )

        #Update Each theta
        for x in range(0,n+1):
            theta[x] = theta[x] - (lr * float(gradients[x]))

    return poly