import numpy as np
import numpy.typing as npt


def gradient_descent(x: npt.NDArray[np.int_], y: npt.NDArray[np.int_]):
    # current values
    m_curr = b_curr = 0
    # tweak number of iterations to get to value
    iterations: int = 1000
    n: int = len(x)
    # rate at which the system adjusts. Bigger number = bigger step, adjust as long as cost decreases with each iteration
    learning_rate: float = 0.08

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr 
        # cost function for mean squared error
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        # partial derivatives
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print(f'm: {m_curr}, b: {b_curr}, cost: {cost}, iteration: {i}')


x: npt.NDArray[np.int_] = np.array([1, 2, 3, 4, 5])
y: npt.NDArray[np.int_] = np.array([5, 7, 9, 11, 13])

gradient_descent(x, y)
