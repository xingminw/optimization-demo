# This is a brief solver for conditional gradient method
# to numerically solve the constrained optimization

# This method is also called the Frank-Wolfe Method
#  and is used to solve the linear constrained problem:
# min z(x), s.t. Ax <= b

import numpy as np
from math import pow, sqrt
from scipy.optimize import linprog
import matplotlib.pyplot as plt


def target_function(x_list=None, start_point=None, end_point=None, alpha=None):
    if x_list is None:
        dimension = len(start_point)
        x_list = np.zeros(dimension).tolist()
        for idx in range(dimension):
            x_list[idx] = (1 - alpha) * start_point[idx] + alpha * end_point[idx]
        # print(alpha, start_point, end_point, x_list)

    y = 4*pow(x_list[0] - 10, 2) + pow(x_list[1] - 4, 2)
    return y


def target_derivative(x_list, method="numerical"):
    """
    take the derivative of the original target function

    :param x_list:
    :param method: "numerical" or analytical
    :return:
    """
    if method == "numerical":
        small_step = 0.0000001
        dx = target_function([x_list[0] + small_step, x_list[1]]) - target_function(x_list)
        dx = dx / small_step
        dy = target_function([x_list[0], x_list[1] + small_step]) - target_function(x_list)
        dy = dy / small_step
    else:
        # customize the derivative function here
        dx = 8 * (x_list[0] - 10)
        dy = 2 * (x_list[1] - 4)
    return [dx, dy]


def conditional_gradient_opt(a_ub, b_ub, start_point, max_iters=100):

    current_point = start_point
    for current_iter in range(max_iters):
        print("<===============Iteration " + str(current_iter) + "===============>")
        print("Currently the point is:", [np.round(val, 3) for val in current_point])
        c = target_derivative(current_point, "analytical")
        print("The current first order derivative is:", [np.round(val, 3) for val in c])

        lp_ans = linprog(c, A_ub=a_ub, b_ub=b_ub)
        direction_end = lp_ans.x
        print("The direction end point is:", [np.round(val, 3) for val in direction_end])

        alpha, opt_point = golden_section_opt(current_point, direction_end)
        print("The step size alpha is", np.round(alpha, 3),
              ", and the current point is", [np.round(val, 3) for val in opt_point])
        current_point = opt_point
        if alpha < 0.000001:
            break


def golden_section_opt(start_point, end_point, min_interval=0.0000000001, max_iters=10000):
    alpha = None
    golden_prop = (sqrt(5) - 1) / 2

    alpha_min = 0
    alpha_max = 1
    left_alpha = golden_prop * alpha_min + (1 - golden_prop) * alpha_max
    right_alpha = (1 - golden_prop) * alpha_min + golden_prop * alpha_max

    current_idx = 0
    while 1:
        current_idx += 1
        if current_idx > max_iters:
            break

        left_val = target_function(start_point=start_point, end_point=end_point, alpha=left_alpha)
        right_val = target_function(start_point=start_point, end_point=end_point, alpha=right_alpha)

        if abs(alpha_max - alpha_min) < min_interval:
            break

        if left_val < right_val:
            alpha = left_alpha
            current_min_val = left_val
            alpha_max = right_alpha
            right_alpha = left_alpha
            left_alpha = golden_prop * alpha_min + (1 - golden_prop) * alpha_max
        else:
            alpha = right_alpha
            current_min_val = right_val
            alpha_min = left_alpha
            left_alpha = right_alpha
            right_alpha = (1 - golden_prop) * alpha_min + golden_prop * alpha_max

    dimension = len(start_point)
    opt_point = np.zeros(dimension).tolist()
    for idx in range(dimension):
        opt_point[idx] = (1 - alpha) * start_point[idx] + alpha * end_point[idx]
    return alpha, opt_point


def conditional_gradient_sample():
    a_ub = [[1, -1], [-1/3, 1], [-1, 0]]
    b_ub = [10, -3, 0]
    start_point = [0, -10]
    conditional_gradient_opt(a_ub, b_ub, start_point)


def numerical_test():
    alpha = np.linspace(0, 1, 101).tolist()
    start_point = [0, -10]
    end_point = [10.5, 0.5]
    val_list = []

    for a in alpha:
        val_list.append(target_function(start_point=start_point, end_point=end_point, alpha=a))

    plt.figure()
    plt.plot(alpha, val_list)
    plt.show()

    print(alpha)


if __name__ == '__main__':
    conditional_gradient_sample()
    # numerical_test()

