#--------------------
# CLASS METHOD
# class person:
#     name = "inam"
#     occupation = "Generative AI developer"
#     def info(self):
#        print(f"my name is {self.name} and my occupation is {self.occupation}")
# a=person()
# b=person()
# b.name = "Tahir"
# b.occupation= "softwere Enginner"
# b.info()
# # print(a.name , a.occupation)
# a.info()
#----------------------------
#CONSTRUCTOR METHOD
#there are tow type of constructor
# 1) PARAMETERIZED CONSTRUCTOR
#2) DEFAULT CONSTRUCTOR
#-------------
# parameterized constructor
class person:
    def __init__ (self, name, occupation):
        print("hey i am a new employee")
        self.name= name
        self.occupation= occupation
    def information(self):
        print(f"My name is {self.name} and my Occupation {self.occupation} ")

a = person("inam", "Generative AI Enginnner")
b = person("neha", "house Wife")
a.information()
b.information()

class Office:
    def __init__ (self, name, occupation, salary):
      print("this is employee list")
      self.name= name
      self.occupation=occupation
      self.salary= salary
    def info(self):
        print(f"this employe name is {self.name} and work is {self.occupation} this worker payment is {self.salary}")
worker1 = Office("inam", "Generative AI Enginner", "5000$")
worker2 = Office("Tahie", "Softwere Enginner", "3000$")
worker3 = Office("Habib", "airTech softwere", "3000$")
worker1.info()
worker2.info()
worker3.info()
#-------------
# default consructor
class animal :
    def __init__ (self):
        print("the crab is the crustaceans animal group")
object1 = animal()
