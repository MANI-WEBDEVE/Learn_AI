class Student :
    universityName = "NAD University"
    def __init__(self, name):
        self.name = name,
        self.GPA = 3

    def infoStudent(self):
        print(f"student name is {self.name} and university name is {self.universityName} ranke of   GPA is {self.GPA}")
student1 = Student("inam");
student1.GPA = 10
student1.infoStudent()
student2 = Student("tahir")
student2.infoStudent()
# example class variable
class Employee:
    class_varaible = 0
    def __init__(self):
        Employee.class_varaible +=1
    def print_class_variable(self):
        print(Employee.class_varaible)
obj1 = Employee()
obj2 = Employee()
obj1.print_class_variable()
obj2.print_class_variable()
# example instance variable
