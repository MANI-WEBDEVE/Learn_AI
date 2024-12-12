#now this topic for lambda
def squre (x, y):
    return x * y
print(squre(3,3))
#now write lambda function
divide = lambda x , y : x / y 
print(int(divide(2,2)))
#and now write a whole sqaure formula
whole = lambda x , y : x*x + y*y + 2*x*y
print(whole(12,12))
#and now write other example
def appl (x, value):
    return 6 + x(value)
 #square statement 
double = lambda x : x * 2 
cube = lambda x: x * x * x
avg = lambda x, y, z:(x + y + z)/3
print(double(3))
print(cube(3))
print(avg(2,2,2))
print(appl(lambda x: x * x,2))
# print(appl(2))