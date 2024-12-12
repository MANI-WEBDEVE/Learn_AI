#example of getters and setters
class MyClass:
    def __init__(self, value):
        # print(self, value)
        self._value= value
    def show (self):
        print(f"value is {self._value}")
#getter example
    @property
    def ten_value(self):
        return 10 / self._value
#setter Example
    @ten_value.setter
    def ten_value(self, new_value):
        self._value = new_value/10
        return 10 / self._value
obj = MyClass(10)
# print(obj.ten_value)
obj.ten_value= 67
obj.show()