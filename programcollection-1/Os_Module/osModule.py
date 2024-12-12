import os
# print(dir(os))
if(not os.path.exists("pro")):#ya statement check karte ha ka folder exist karta ha ya nahi 
    result = os.mkdir("pro")
for i in range(0 , 100):
    os.mkdir(f"pro/tutoral{i+1}") #ya function create folder kar ta ha 
       