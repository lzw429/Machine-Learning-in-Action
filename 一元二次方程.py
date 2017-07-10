import math


def quadratic(a, b, c):
    if b * b - 4 * a * c >= 0:
        return (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a), (-b - math.sqrt(b * b - 4 * a * c)) / (2 * a)
    else:
        res1 = str(-b / (2 * a)) + '+' + str(math.sqrt(abs(b * b - 4 * a * c))) + 'i'
        res2 = str(-b / (2 * a)) + '-' + str(math.sqrt(abs(b * b - 4 * a * c))) + 'i'
        return res1 + res2


print("Please input a, b and c")
a = input()
b = input()
c = input()
print(quadratic(a, b, c))
