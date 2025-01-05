#MAP
#map filter apply karna sa phle khuch example dekhel lata ha
lists= [1,2,6,4,5,3]
def cube(x):
    return x*x*x
newlist = []
for list in lists:
    newlist.append(cube(list))
print(newlist)
#ya jo uppper kam kara ha kitna haritate ha
#is liya hum python ka inbuild function ka istamal karta ha
newl = list(map(lambda x : x*x*x, lists))
print(newl)

#second is filter is kam name sa maloome parta ha ka ya filter karta ha kisi number ko
item = [1,3,2,3,4,5,6,5,4,3,3]
def filter_function(a):
    return a > 3
fill = list(filter(filter_function, item))
print(fill)

#now use reduce function
#hum na MAP, FILTER istamal kara lakin unko import nahi kiya lakin reduce ko import karana parta ha
from functools import reduce
num = [1,2,34,5,3]
def sum(x ,y):
    return x+y% 2 
even = reduce(sum , num)
print(even)