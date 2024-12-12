string = "12";
# string = "1";
number = 7;
#this line string convert number with help of int() function
string_number= int(string);
#-----------------------------------------------------------
sum = number +string_number
print("the sum of both the number is ", sum)

#example of Implicit type casting
a = 9;
print(type(a));
b= 3.0
print(type(b));
c = a + b;
print(c)
print(type(c))
