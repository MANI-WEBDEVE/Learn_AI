marks = [122,121,133,131,111,90,89]
index = 0
for mark in marks :
    if(mark == 90):
        print( f" {index}shamoni you")
    if(mark == 133):
        print(f" {index} good performance but hard working")
    index += 1

#ys jo top per progarm likha ha thora defficult ha
#lakin hum {Emurate function} ka istamal kar kye is ko asan tarika sa  likhe sakta ha
roll = [12 ,19,21,27,23,34,36,46,67,77]
#phele hum index represent kar raha tha jisa ka kuch asa
#{ index = 0 }
for index, rollno in enumerate(roll):
    if(rollno == 34):
        print(f"good and daily student {index}")
    if(rollno == 67):
        print(f"like noise man {index}")
    if(rollno == 12):
        print(f"smart boy like INAM {index}")

#example is now
fruit = ["mango", "banana", "oranges", "Graphs"]
for index, phale in enumerate(fruit, start=1):
    print(f"{index} this is gym power {phale}")
