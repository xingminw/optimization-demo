# this code is to find the zero point of a function using bi-section method
# the target function here must be monotonically increased or decreased


def target_function(x):
    y = pow(x-4, 1)
    return y


def bi_section_solver(range_min, range_max, interval_min=0.001, tolerance=0.0001):
    while 1:
        left_val = target_function(range_min)
        right_val = target_function(range_max)

        if abs(range_min - range_max) < interval_min:
            if min(abs(left_val), abs(right_val)) < tolerance:
                ans_x = range_max
                solution_state = "Found"
            else:
                ans_x = range_max
                solution_state = "Not reach yet"
            break

        if abs(left_val) < tolerance:
            ans_x = range_min
            solution_state = "Found"
            break

        if abs(right_val) < tolerance:
            ans_x = range_max
            solution_state = "Found"
            break

        if left_val * right_val > 0:
            ans_x = None
            solution_state = "No solution"
            break

        middle_x = (range_min + range_max) / 2
        mid_val = target_function(middle_x)

        if abs(mid_val) < tolerance:
            ans_x = middle_x
            solution_state = "Found"
            break
        else:
            if mid_val * left_val < 0:
                range_max = middle_x
            else:
                range_min = middle_x
    return ans_x, solution_state


def bi_section_sample():
    ans, state = bi_section_solver(-20, 500)
    print(ans, state)


if __name__ == '__main__':
    bi_section_sample()


