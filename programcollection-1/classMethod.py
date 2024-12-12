class ExampleEmployee:
    company = "Pixel"
    def show(self):
        print(f"the name is {self.name} and company name is {self.company}")

#todo:you do add class method decorator you doon`t change real time company name
    @classmethod
    def changeCompany(cls, newCompany):
        cls.company = newCompany

e1 = ExampleEmployee()
e1.name = "Muhammad Inam"
e1.changeCompany("Tesla")
print(ExampleEmployee.company)
e1.show()
e1.name = "muhammad furkan"
e1.changeCompany("Google")
e1.show()
print(ExampleEmployee.company)