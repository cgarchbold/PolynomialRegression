import numpy as np

def MSE(X,Y, poly):
    """
    Calculate the Mean Squared Error (MSE) poly(X) and Y
    """
    return np.sum( ( Y - poly(X) ) ** 2 ) / len(X)

def MSEderiv(X, Y, poly, n):
    """
    Calculate the partial derivative of the MSE on parameter order n
    """
    return np.sum( ( poly(X) - Y ) * (X ** n) ) / len(X)

def poly_reg(X, Y, n, lr, iterations, batch_size):
    """
    Regression Function - Preform Polynomial regression with order n polynomial, return the best polynomial
    """
    theta = np.random.rand(n+1)

    for i in range(iterations):
        
        poly = np.poly1d(theta)

        #Evaluate Total Loss
        cost = MSE(X,Y, poly)
        if(i % 100 == 0):
            print(f"Epoch[{i}] : MSE = {cost}")
        
        #Select a random mini-batch
        random_indices = np.random.choice(len(X), size=batch_size, replace=False)
        x = X[random_indices]
        y = Y[random_indices]

        gradients = []
        #For each theta, calculate gradient
        for order in reversed(range(0,n+1)):
            gradients.append( MSEderiv(x,y,poly,order) )

        #Update Each theta
        for x in range(0,n+1):
            theta[x] = theta[x] - (lr * float(gradients[x]))

    return poly
