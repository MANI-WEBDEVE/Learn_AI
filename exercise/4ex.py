import random 
comp = random.randint(0,2)

user = int(input("0 for snake, 1 for water and 2 for Gun  "))


def compare (comp, user):
    if(comp == user):
        return 0
    if(comp == 0 and user == 1):
        return -1
    if(comp == 1 and user ==2):
       return -1
    if(comp == 2 and user == 0):
        return -1
    
score =compare(comp, user)
print(f"User {user}")
print(f"computer {comp}")
##################
if(score == 0):
    print("It`s Draw")
elif(score == -1):
    print("You Lose")
else:
    print("You Win Player")