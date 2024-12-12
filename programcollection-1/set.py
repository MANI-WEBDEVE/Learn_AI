allType = {56,56,78,5,78,"inam", "saad", True,"inam"}
print(allType)
print(type(allType))
#but set jab khali ho tu set type nahi hote balkai dictinar type hotie ha
set1= {}
print(type(set1))
#or
set2 = set()
print(type(set2))
#using for loop
info ={"inam", 42201,-9845231 ,-1, True, "inma"}
for information in info:
    print(information)

#set method
cities = {"karachi", "Lahore", "punjab", "multan"}
cities2 = {"Blochistan", "hydrabad", " rawalpindi", }
cities3 = cities.union(cities2)
print(cities3)
#--------------------
cities = {"karachi", "Lahore", "punjab", "multan"}
cities2 = {"Blochistan", "hydrabad", " rawalpindi", }
cities.update(cities2)
print(cities)
#----------------------
numSeq = {12,13, 10, 19, 17, 67}
numSeq2 = {10,21,31,45,43,12,14,19}
numSeq3 = numSeq.intersection(numSeq2)
#---------------
print(numSeq3)
numSeq = {12,13, 10, 19, 17, 67}
numSeq2 = {10,21,31,45,43,12,14,19}
numSeq2.intersection_update(numSeq)
print(numSeq2)
#------------------
#symetric difference
city = { "bhpal", "banglore", "mumbai", "karachi", "islamabad"}
city2 = {"bhpal", "karachi", "multan"}
city3 = city.symmetric_difference(city2)
print(city3)
#---------------
city = { "bhpal", "banglore", "mumbai", "karachi", "islamabad"}
city2 = {"bhpal", "karachi", "multan"}
city.symmetric_difference_update(city2)
print(city)
#---------------
name = {"inam", "tahir","ayan", "saad"}
name2 = {"tahir", "sharjeel", "ayan"}
name3 = name.difference(name2)
print(name3)
#-----------------
name = {"inam", "tahir","ayan", "saad"}
name2 = {"tahir", "sharjeel", "ayan"}
name.difference_update(name2)
print(name)
#---------------------
muslim = {"pakistan", "iran", "turkey", ""}
nonmuslim = {"America", "canada", "Australia"} 
print(muslim.isdisjoint(nonmuslim))
#-------------
muslim = {"pakistan", "iran", "turkey", "Qatatr"}
nonmuslim = {"America", "canada", "Australia"} 
print(muslim.issuperset(nonmuslim))
othermuslim = { "iran" }
print(muslim.issuperset(othermuslim))
#------------
hacker ={ "whitehacker", "blackhacker"}
hacker2 = {"grayhacekr", "greenhacker", "blackhacke"}
hacker3= hacker2.issubset(hacker)
print(hacker3)
#----------------
freind = {"inam", "saad", 'sharjeel', "rohan"}
freind.add("ayan")
print(freind)
#----------
language = {"english ", "urdu", "chines"}
language2 ={"punjabi", "balochi","spanies"}
language.update(language2)
print(language)
#-------------------
language3 = {"english ", "urdu", "chines"}
language3.remove("urdu")
print(language3)
#--------------
language4 = {"english ", "urdu", "chines","punjabi", "balochi","spanies"}
lang = language4.pop()
print(language4)
print(lang)
#---------------------
gamer = {"technoGamer", "ujwaalGamer", "triggersGamer"}
del gamer
# print(gamer)
#------------
gamer = {"technoGamer", "ujwaalGamer", "triggersGamer"}
gamer.clear()
print(gamer)
