# Qusetion No 1

Accept two numbe and printthe greates between them

# solution

````a = int(input("Enter The First Number"))
b = int(input("Enter The Second Number"))

`if(a > b ):
    print(f"{a} value is greater then {b} ")
elif(b > a):
     print(f"{b} value is greater then {a}")
else:
     print(f"{a} both are equal {b}")```
````

# Question No 2

Accept the gender from the user as character and print the repective greeting message

# Example

GOOD MORNING SIR (ON THE BASICS OF GENDER)

# Solution

````gender = input("Provide a gender F and M  ")
if ( gender == "F" or "f"):
    print("Have a nice trip MAM")
elif (gender == "M" or "m"):
    print("have a nice trip SIR")
else:
    print("Not allow SheMale")```
````

#

# Qusetion No 3

Accept two integer and check whether it is an even number or Odd

#

# Solution

````first = int(input("enter the first number "))
if (first % 2 == 0):
    print(f"this is even number {first}")
elif(first % 2 != 0):
    print(f"this number is odd {first}")
else:
    print(None)```
````

#

# Question No 4

Accept name and age from the user check if the user is a valid voter or not

# Solution

````name = input("please enter your name ")
age = int(input("enter your age "))
if(age >= 18 ):
    print(f"hello MR {name} your are a valid voter")
elif(age < 18 and age > 0):
    print(f"hello mr {name} sorry are you not valid voter")
else:
    print(f"hello mr {name} you are enter wrong age")```
````

#

# Question No 5

Accept a year and check if it a leap year or not

#

# Solution

```year = int(input("enter the year and find leap "))
if (year % 4 == 0 and year % 100 != 0):
    print(f"this year {year} is leap year")
elif(year % 100 == 0 and year % 400 == 0):
    print(f"this {year } is  a leap year")
else:
    print("its is not leap year")
```

#

# Question No 6

Accept an English Aplhabet from user an check if it is a consonant or avowel

#

# Solution

```
alpha = input("enter the aplhabate and check this aplhabate vowel to and conconent ")
if(alpha in "aeiouAEIOU"):
    print(f"this is vowel  {alpha}")
else:
    print(f"this is not a vowel {alpha}")
```
