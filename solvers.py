
class Solvers:
    def __init__(self, initial_point, gradient, 
                step_size = 0.05, error = 0, iterations = None):
        self.initial_point = initial_point
        self.gradient = gradient
        self.step_size = step_size
        self.error= error
        self.iterations = iterations
        
    def steepest_descent(self):
        """
        Following code computes the steepest descent method of a 
        fixed function.
        """
        
        aux_point = self.initial_point
        t = 0
        while (self.gradient(aux_point)[0]**2 + self.gradient(aux_point)[1]**2)**0.5 > self.error and t < self.iterations:
            
            print(f'aux_point: {aux_point}, gradient = {self.gradient(aux_point)}')
            gradient_aux = self.gradient(aux_point)
            aux_point[0] = aux_point[0] - self.step_size * gradient_aux[0]
            aux_point[1] = aux_point[1] - self.step_size * gradient_aux[1]
            print(aux_point[1])
            t += 1
        return aux_point

if __name__ == "__main__":
    #Lets implement an example of how to use the steepest descent method, assuming that
    #the function will be: (1.5*x**2 + 0.5*y**2 - x*y - 2*x)
    #1. Defining the initial point:
    initial_point_ = [-2,4]
    #2. Defining gradient function:
    
    def gradient_1(x):
        return [3*x[0]-x[1]-2, x[1]-x[0]]
    
    #3. Assuming that the step size  is 0.5, we initialize the Solvers class
    
    solver = Solvers(initial_point_, gradient=gradient_1, step_size=0.5, iterations=7)
    print(solver.steepest_descent())
    