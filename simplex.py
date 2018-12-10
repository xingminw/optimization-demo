# code for simplex method
# print all the details of the simplex method

import numpy as np


def simplex_method(matrix_a, matrix_c, matrix_b, basic_vars,
                   non_basic_vars, max_iters=100, print_detail=True):
    """
    This is to solve the LP using the simplex method
    print the detail when conducting the simplex method

    the standard form of LP is:
    min c^T*x, s.t. Ax = b, x >= 0
    basic vars and non basic vars have to be initiated !!!

    :param matrix_a:
    :param matrix_c:
    :param matrix_b:
    :param basic_vars:
    :param non_basic_vars:
    :param max_iters:
    :param print_detail:
    :return:
    """
    solution_state = None

    for idx in range(max_iters):
        if print_detail:
            print("===================================================")
            print("Iteration Number", idx)

            print("The current basic variables are:",
                  ",".join(["x" + str(val + 1) for val in basic_vars]))
            print("The current non-basic variables are:",
                  ",".join(["x" + str(val + 1) for val in non_basic_vars]))

        matrix_basis = matrix_a[:, basic_vars]
        current_vars = np.dot(np.linalg.inv(matrix_basis), matrix_b)

        if print_detail:
            print("The current values of variables are:", current_vars.T[0])

        c_basic = matrix_c[basic_vars, :]
        current_obj = np.dot(np.transpose(c_basic), current_vars)[0, 0]

        if print_detail:
            print("The current value objective function is:", current_obj)

        c_non_basic = matrix_c[non_basic_vars, :]
        matrix_a_non_basic = matrix_a[:, non_basic_vars]

        temp_val1 = np.dot(np.transpose(c_basic), np.linalg.inv(matrix_basis))
        temp_val2 = np.dot(temp_val1, matrix_a_non_basic)

        obj_inc = c_non_basic.T - temp_val2
        if print_detail:
            print("The current change rate of objective functions is:", obj_inc[0])

        temp_index = int(np.argmin(obj_inc[0]))
        obj_inc_min = obj_inc[0][temp_index]

        if obj_inc_min > 0:
            if print_detail:
                print("This is the optimal solution")
            solution_state = "optimal solution"
            break

        non_basic_index = non_basic_vars[temp_index]
        del non_basic_vars[temp_index]
        basic_vars.append(non_basic_index)

        if print_detail:
            print("The chosen non-basic index of variables is", "x" + str(non_basic_index + 1))

        jth_matrix_a = matrix_a[:, non_basic_index]
        direction_b = - np.dot(np.linalg.inv(matrix_basis), jth_matrix_a)

        candidate_step_size_list = []
        candidate_basic_idx = []
        for direction_idx in range(np.shape(direction_b)[0]):
            local_direction_b = direction_b[direction_idx]
            local_val = current_vars[direction_idx]
            if local_direction_b < 0:
                candidate_step_size_list.append(- (local_val / local_direction_b)[0])
                candidate_basic_idx.append(direction_idx)

        if len(candidate_step_size_list) == 0:
            if print_detail:
                print("This problem is unbounded!")
            solution_state = "Unbounded"
            break

        temp_index = candidate_basic_idx[int(np.argmin(candidate_step_size_list))]
        chosen_basic_index = basic_vars[temp_index]
        del basic_vars[temp_index]
        non_basic_vars.append(chosen_basic_index)

        step_size = np.min(candidate_step_size_list)
        if print_detail:
            print("The chosen basic index of variables is", "x" + str(chosen_basic_index + 1))
            print("The step size is", step_size)

        basic_vars = np.sort(basic_vars).tolist()
        non_basic_vars = np.sort(non_basic_vars).tolist()
        solution_state = "Maximum Iter"
    return solution_state


def run_simplex_sample():
    matrix_a = np.array([[1, -2, 1, 0, 0], [-2, 1, 0, 1, 0], [5, 3, 0, 0, 1]])
    matrix_c = np.array([[-1], [-3], [0], [0], [0]])
    matrix_b = np.array([[0], [4], [15]])

    basic_vars = [2, 3, 4]
    non_basic_vars = [0, 1]
    simplex_method(matrix_a, matrix_c, matrix_b, basic_vars, non_basic_vars, print_detail=True)


if __name__ == '__main__':
    run_simplex_sample()
