# #table formate method
# user = int(input("enter the integer"))
# print(f"multiplication table of {user} ")
# try:
#     for i in range(1 ,11):
#         print(f"{user} X {i} = {user * i}")
# except:
#     print("sorry Not Found")

# try:
#     num = int(input("enter the number"))
#     a = [2,4,4]
#     print(a[num])
# except ValueError:
#     print("this is invalid syntex")
# except IndexError:
#     print("this is invalid index")

# finally clause 
# user = int(input("enter the number"))

# try:
#     if( user > 8 or user < 18):
#       print('this is login admin panel')
#     print(f"this is not login admin panel {user}")
# except ValueError:
#     print(f"this is not correct password {user}")
# finally:
#     print("this is don")

#custom Error 
a = int(input("Enter any value between 5 and 9"))

if(a<5  or a>9):
  raise  ValueError("Value should be between 5 and 9")
     