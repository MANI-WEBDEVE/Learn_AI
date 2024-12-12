class Student:
    def __init__(self):
        self._name = "Harry"

    def _funName(self):  # protected method
        return "CodeWithHarry"


class Subject(Student):  # inherited class
    pass


obj = Student()
obj1 = Subject()
print(dir(obj))

# calling by object of Student class
print(obj._name)
print(obj._funName())
# calling by object of Subject class
print(obj1._name)
print(obj1._funName())
###############################
class myClass:
    def  __init__(self):
        self._nonMangling_attribute = "Muhammad Inam"
        self._mangled_attribute = "Muhammad Tahir"
my_object = myClass
print(my_object._nonMangling_attribute)
print(my_object.__mangled_attribute)
print(myClass._my_object.__mangled_attribute)
#################################