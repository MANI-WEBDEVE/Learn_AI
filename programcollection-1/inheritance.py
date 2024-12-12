class Employee:
    def __init__(self, name, id):
        self.name= name
        self.id= id

    def showEmployeDetaile(self):
        print(f"this is employe id is {self.id} and name is {self.name}")
class progarmmer(Employee):
    def __init__(self, lang):
        self.lang= lang
    def showLanguage(self):
        print(f"the default language is {self.lang}")
FirstEmployee = progarmmer("Java")
e1 = Employee("inma", 1212)
e1.showEmployeDetaile()
# SecondEmployee= progarmmer ("TAHIR", 124)
# FirstEmployee.showEmployeDetaile()
FirstEmployee.showLanguage()
# SecondEmployee.showLanguage("java")