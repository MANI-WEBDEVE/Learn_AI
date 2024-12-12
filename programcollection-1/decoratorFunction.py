# Decorated function
import logging
print(logging)
# def greet(fx):
#     def mfx():
#         print("Good Morning")
#         fx()
#         print("this world is primer")   
#     return mfx  
# @greet 
# def hello():
#     print("hello world")
# hello()
#--------------
# def example(function):
#     def modified_function(*arugument, **karguments):
#         print("good Maths")
#         function(*arugument, **karguments)
#         print("i hope solve this question")
#     return modified_function

# # @example
# def square (a, b):
#     print( a*b)
# example(square)(2,3)     
#------------
#PRACTICAL EXAMPLE
def log_function_call(function):
    def decorated(*args, **kwargs):
        logging.info(f"Calling{function.__name__} with args = {args} and kwargs= {kwargs}")
        print(logging.info)
        print(decorated)
        #result name ka variable bana na ha is ma bhi do variable pass karna ha 
        result = function(*args, **kwargs)
        logging.info(f"{function.__name__} and resultr = {result}")
        return result
    return decorated
@log_function_call
def my_function_add(a, b):
    return a+ b
my_function_add(2, 9)
# print(my_function_add)