# Example of branch and bound for integer linear programming
import numpy as np
from copy import deepcopy
from scipy.optimize import linprog
from math import floor, ceil


# Official guide for linprog:
# scipy.optimize.linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
#                        bounds=None, method='simplex', callback=None, options=None)[source]
# Minimize:     c^T * x
# Subject to:   A_ub * x <= b_ub
#               A_eq * x == b_eq

def ilp_algorithm(c, a_ub=None, b_ub=None, a_eq=None, b_eq=None):
    # first calculate the relaxation lp problem
    lp_ans = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq)
    obj_val = lp_ans["fun"]
    ans_x = lp_ans["x"].tolist()
    sol_status = lp_ans["status"]

    if sol_status > 1:
        print("Not feasible or unbounded!")
        exit()

    print("The relaxation LP problem:")
    print("Objective function", np.round(obj_val, 3), "Current x",
          [np.round(val, 2) for val in ans_x], "obj", np.round(obj_val, 3))

    upper_bound = 100000000
    current_best_dict = {"ans": None, "obj": None}
    init_branch_dict = {"a_ub": a_ub, "a_eq": a_eq, "b_ub": b_ub,
                        "b_eq": b_eq, "ans": ans_x, "obj": obj_val}

    branch_layers_list = [{"init": init_branch_dict}]
    max_layers = 10

    print()
    print("Start to iteration for branch and bound")
    for idx in range(max_layers):
        # this iteration is to make every variable a integer

        local_layer_dict = branch_layers_list[idx]
        update_layer_dict = {}

        if len(local_layer_dict.keys()) == 0:
            break

        for local_key in local_layer_dict.keys():
            local_node = local_layer_dict[local_key]
            local_a_ub = local_node["a_ub"]
            local_b_ub = local_node["b_ub"]
            local_a_eq = local_node["a_eq"]
            local_b_eq = local_node["b_eq"]
            local_ans = local_node["ans"]
            local_obj = local_node["obj"]

            print("<==============================>")
            print("local node key:", local_key)
            print("local ans", [np.round(val, 3) for val in local_ans], "local obj", np.round(local_obj, 3))

            if local_obj > upper_bound:
                break

            non_integer_idx = return_non_integer_idx(local_ans)

            if non_integer_idx is None:
                if local_obj < upper_bound:
                    upper_bound = local_obj
                    current_best_dict["ans"] = local_ans
                    current_best_dict["obj"] = local_obj
                break

            local_non_integer = local_ans[non_integer_idx]

            # branch 1
            branch_key = local_key + "-x" + str(non_integer_idx + 1) + ">" + str(int(local_non_integer) + 1)
            extra_b_ub = - (int(local_non_integer) + 1)
            extra_a_ub = np.zeros(len(local_ans)).tolist()
            extra_a_ub[non_integer_idx] = -1

            if local_a_ub is None:
                update_a_ub = []
                update_b_ub = []
            else:
                update_a_ub = deepcopy(local_a_ub)
                update_b_ub = deepcopy(local_b_ub)

            update_a_ub.append(extra_a_ub)
            update_b_ub.append(extra_b_ub)

            lp_ans = linprog(c, A_ub=update_a_ub, b_ub=update_b_ub, A_eq=local_a_eq, b_eq=local_b_eq)
            local_update_ans = lp_ans.x
            local_update_obj = lp_ans.fun
            local_update_status = lp_ans.status

            if local_update_status == 0:
                temp_branch_dict = {"a_ub": update_a_ub, "a_eq": local_a_eq, "b_ub": update_b_ub,
                                    "b_eq": local_b_eq, "ans": local_update_ans, "obj": local_update_obj}
                update_layer_dict[branch_key] = temp_branch_dict

            # branch 2
            branch_key = local_key + "-x" + str(non_integer_idx + 1) + "<" + str(int(local_non_integer))
            extra_b_ub = int(local_non_integer)
            extra_a_ub = np.zeros(len(local_ans)).tolist()
            extra_a_ub[non_integer_idx] = 1

            if local_a_ub is None:
                update_a_ub = []
                update_b_ub = []
            else:
                update_a_ub = deepcopy(local_a_ub)
                update_b_ub = deepcopy(local_b_ub)

            update_a_ub.append(extra_a_ub)
            update_b_ub.append(extra_b_ub)

            lp_ans = linprog(c, A_ub=update_a_ub, b_ub=update_b_ub, A_eq=local_a_eq, b_eq=local_b_eq)
            local_update_ans = lp_ans.x
            local_update_obj = lp_ans.fun
            local_update_status = lp_ans.status

            if local_update_status == 0:
                temp_branch_dict = {"a_ub": update_a_ub, "a_eq": local_a_eq, "b_ub": update_b_ub,
                                    "b_eq": local_b_eq, "ans": local_update_ans, "obj": local_update_obj}
                update_layer_dict[branch_key] = temp_branch_dict

        branch_layers_list.append(update_layer_dict)
    print("Complete the branch and bound")

    print()
    print("The result of the integer programming is:")
    print("best ans", current_best_dict["ans"], "min fun", current_best_dict["obj"])


def return_non_integer_idx(temp_list, tolerance=0.0000001):
    for idx in range(len(temp_list)):
        local_val = temp_list[idx]
        dis_to_floor = local_val - floor(local_val)
        dis_to_ceil = ceil(local_val) - local_val

        local_residual = min(dis_to_ceil, dis_to_floor)
        if local_residual > tolerance:
            return idx
    return None


def ilp_sample():
    c = [-9, -5, -6, -4]
    # a_ub = [[6, 3, 5, 2], [0, 0, 1, 1], [-1, 0, 1, 0], [0, -1, 0, 1]]
    # b_ub = [10, 1, 0, 0]

    a_ub = [[6, 3, 5, 2], [0, 0, 1, 1], [-1, 0, 1, 0], [0, -1, 0, 1],
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    b_ub = [10, 1, 0, 0, 1, 1, 1, 1]
    ilp_algorithm(c, a_ub=a_ub, b_ub=b_ub)


if __name__ == '__main__':
    ilp_sample()
