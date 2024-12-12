#factorial function
#factorial 6*5*4*3*2*1
def factorial(number):
    if(number == 0 or number == 1):
        return 1
    else:
        return number * factorial(number -1)
print(factorial(1))

#fibonnaci sequence programe build
#f =0
#f =1
#f(2) = f(1) + f(0)
#f(n) = f(n-1) + f(n-2)
def fibinnaci(num):
    if(num == 0 or num == 1):
        return 1
    else:
        return fibinnaci(num - 1) + fibinnaci(num -2)
print(fibinnaci(5))