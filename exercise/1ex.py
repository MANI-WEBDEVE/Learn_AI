import time
t = time.strftime("%H,%M,%S")
# hour = int(time.strftime("%H"))
hour = int(input("Enter the hourse"))
print(hour)
if(hour>=0 and hour<12):
    print("good morning sir") 
elif(hour>=12 and hour<16):
    print("afternoon sir")
elif(hour>=17 and hour<19):
    print("good evening sir")
elif(hour>=20 and hour<24):
    print("good Night sir")
    #this project is very simple but leearn the advanced concept and come back this projects better then projects 