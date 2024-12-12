x_axis = 12;#its global variable
print(x_axis)
def local_function():
     y_axis=11
     print(f"this is X_axis{x_axis}")
     print(f"this is y_axis{y_axis}")
    
local_function()
print(f"this is X_axis{x_axis}")
# print(f"this is y_axis{y_axis}")
