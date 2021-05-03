import numpy as np
import math 

class Solver:
    def __init__(self,  method = 'gradient', initial_point = np.array([0,0]), gradient = None, 
                step_size = 0.05, error = 0.05, iterations = 100, finite_differences = False,
                hessian_matrix = None, function = None):
        
        #-----------------public methods------------------
        self.method = method
        self.initial_point = initial_point
        self.gradient = gradient
        self.step_size = step_size
        self.error= error
        self.iterations = iterations
        self.finite_differences = finite_differences
        self.hessian_matrix = hessian_matrix
        self.function = function
    
    #-----------------public methods------------------
    def solve (self):
        """
        Solve the minimization problem using the given method.
        
        Return
        ----------
            solution point.
        """
        if self.method is None:
            raise TypeError
        else:
            if self.method == 'gradient' and not self.finite_differences:
                return self.__steepest_descent()
            if self.method == 'gradient' and self.finite_differences:
                return self.__steepest_descent_finite()
            if self.method == 'newton':
                return self.__newton()
            
    
    #-------------------Private methods---------------
    def __steepest_descent(self):
        """
        Following code computes the steepest descent method of a 
        fixed function.
        
        Returns the nearest solution point (list) under fixed initial conditions
        """
        print("------------Solution by Gradient Descent method------------")
        aux_point = self.initial_point.copy()
        t = 0
        gradient_aux = self.gradient(aux_point)
        while math.sqrt(np.dot(gradient_aux,gradient_aux)) > self.error and t < self.iterations:
            print(f'Iteration {t} ; gradient: {gradient_aux}; point: {aux_point} ')
            aux_point = aux_point - np.dot(self.step_size, gradient_aux)
            t += 1
            gradient_aux = self.gradient(aux_point)
            
        return aux_point
    
    def __steepest_descent_finite(self):
        """
        Following function computes the steepest descent method of a 
        fixed function using finite differences.
        
        Returns the nearest solution point (list) under fixed initial conditions
        """
        print("------------Solution by Gradient Descent with Finite Differences method------------")
        
        aux_point = self.initial_point.copy()
        t = 0
        gradient_aux = self.__finite_gradient(aux_point)
        
        while t < self.iterations:
            print(f'Iteration {t} ; gradient: {gradient_aux}; point: {aux_point} ')
            aux_point = aux_point - np.dot(self.step_size, gradient_aux)
            t += 1
            gradient_aux = self.__finite_gradient(aux_point)
        return aux_point
    
    def __finite_gradient(self, x):
        """
        a naive implementation of numerical gradient of f at x
        - f should be a function that takes a single argument
        - x is the point (numpy array) to evaluate the gradient at
        https://github.com/cs231n/cs231n.github.io/blob/master/optimization-1.md#numerical
        """
        
        f = self.function
        fx = f(x) # evaluate function value at original point
        grad = np.zeros(x.shape)
        h = 0.05
        
        # iterate over all indexes in x
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        
        while not it.finished:

            # evaluate function at x+h
            ix = it.multi_index
            old_value = x[ix]
            x[ix] = old_value + h # increment by h
            fxh = f(x) # evalute f(x + h)
            x[ix] = old_value # restore to previous value (very important!)

            # compute the partial derivative
            grad[ix] = (fxh - fx) / h # the slope
            it.iternext() # step to next dimension

        return grad
    def __newton(self):
        """
        Following function computes the Newton method.
        
        Returns the nearest solution point (list) under fixed initial conditions
        """
        print("------------Solution by Newton method------------")
        
        aux_point = self.initial_point.copy()
        t = 0
        gradient_aux = self.gradient(aux_point) #evaluating point in the Gradient
        hessian_aux = self.hessian_matrix(aux_point) #evaluating point in the Gessian Matrix
        if np.linalg.det(hessian_aux):
            inverse_hessian = np.linalg.inv(hessian_aux)
            while math.sqrt(np.dot(gradient_aux,gradient_aux)) > self.error and t < self.iterations:
                print(f'Iteration {t} ; gradient: {gradient_aux};  point: {aux_point} ; \n hessian: {hessian_aux}')
                hessian_aux = self.hessian_matrix(aux_point) #evaluating point in the Gessian Matrix
                inverse_hessian = np.linalg.inv(hessian_aux)
                aux_point = aux_point - np.dot(inverse_hessian,gradient_aux )
                t += 1
                gradient_aux = self.gradient(aux_point)
        return aux_point


if __name__ == "__main__":
    #----------------------------------EXAMPLE OF USE----------------------------------
    #Lets implement an example of how to use the steepest descent method, assuming that
    #the function will be: (1.5*x**2 + 0.5*y**2 - x*y - 2*x)
    #1. Defining the initial point:
    initial_point_ = np.array([-2,4])
    #2. Defining function, gradient and hessian functions:
    
    def gradient_1(x):
        return [3*x[0]-x[1]-2, x[1]-x[0]]
    def hessian_1(x):
        hessian_matrix = np.array([[3, -1],
                                   [-1,1]])
        return hessian_matrix
    def function_1(x):
        return 1.5*x[0]**2 + 0.5*x[1]**2-x[0]*x[1]-2*x[0]
    #3. Assuming that the step size  is 0.5, we instanciate the Solver class
    
    #gradient descent
    solver = Solver(method='gradient',
                    initial_point=initial_point_,
                    gradient=gradient_1,
                    step_size=0.01,
                    iterations=1000)
    print(solver.solve())

    #gradient descent with finite differences
    solver = Solver(method='gradient',
                    initial_point=initial_point_,
                    function=function_1,
                    step_size=0.01,
                    finite_differences=True,
                    iterations=1000)
    print(solver.solve())
    
    #With Newton Method
    solver = Solver(method='newton',
                    initial_point=initial_point_,
                    gradient=gradient_1,
                    hessian_matrix=hessian_1,
                    step_size=0.01,
                    iterations=1000)
    print(solver.solve())
    

    