# linear programming using solver from scipy.optimize
from scipy.optimize import linprog


# Official guide for linprog:
# scipy.optimize.linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
#                        bounds=None, method='simplex', callback=None, options=None)[source]
# Minimize:     c^T * x
# Subject to:   A_ub * x <= b_ub
#               A_eq * x == b_eq


def lp_example():
    c = [-1, -3, 0, 0, 0]
    a_eq = [[1, -2, 1, 0, 0], [-2, 1, 0, 1, 0], [5, 3, 0, 0, 1]]
    b_eq = [0, 4, 15]

    ans = linprog(c, A_eq=a_eq, b_eq=b_eq)
    print(ans)


if __name__ == '__main__':
    lp_example()



