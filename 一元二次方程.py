import math
def quadratic(a,b,c):
    if b*b-4*a*c>=0:
        return (-b+math.sqrt(b*b-4*a*c))/(2*a),(-b-math.sqrt(b*b-4*a*c))/(2*a)
    else:
        return 
a=input("a=")
b=input("b=")
c=input("c=")
print(quadratic(a,b,c))