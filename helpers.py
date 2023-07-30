import math


def exponential_decrease(current_value, upper_limit):
    k = -math.log(upper_limit) / upper_limit
    new_value = math.exp(-k * current_value)
    if current_value > upper_limit:
        return upper_limit
    return new_value


def invert(current_value, upper_limit):
    new_value = (current_value * -1) + upper_limit
    if current_value > upper_limit:
        return upper_limit
    return new_value
