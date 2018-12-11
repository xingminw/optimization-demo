# This code is to find the min val of a one dimension constrained convex function
from math import sqrt


def target_function(x):
    y = (x - 2) * (x - 10)
    return y


def golden_section(x_min, x_max, min_interval=0.000001, max_iters=10000):
    golden_prop = (sqrt(5) - 1) / 2

    current_min_val = None
    ans_x = None

    left_x = golden_prop * x_min + (1 - golden_prop) * x_max
    right_x = (1 - golden_prop) * x_min + golden_prop * x_max

    current_idx = 0
    while 1:
        current_idx += 1
        if current_idx > max_iters:
            break

        left_val = target_function(left_x)
        right_val = target_function(right_x)

        if abs(left_x - right_x) < min_interval:
            break

        if left_val < right_val:
            ans_x = left_x
            current_min_val = left_val
            x_max = right_x
            right_x = left_x
            left_x = golden_prop * x_min + (1 - golden_prop) * x_max
        else:
            ans_x = right_x
            current_min_val = right_val
            x_min = left_x
            left_x = right_x
            right_x = (1 - golden_prop) * x_min + golden_prop * x_max
    print(current_idx)
    return ans_x, current_min_val


def golden_section_sample():
    ans_x, val = golden_section(-10, 30)
    print(ans_x, val)


if __name__ == '__main__':
    golden_section_sample()




