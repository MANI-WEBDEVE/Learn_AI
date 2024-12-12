animals =["lion", "tiger", "deer", "elephante", "zabra"]
print(type(animals))
print(animals)
print(animals[0])
print(animals[4])
print(animals[2])
#now animals find negative indexs
print(4)
#if statement
if "lion" in animals:
    print("yes this one present")
else:
    print("sorry Not Found");
#list comprehension
names=["mani", "hirat", "daas", "bibah"]
namewith_o = [item for item in names if "i" in item]
print(namewith_o)
# example
# number_string =[90, 12, 12, 17, 90, 89, 78,78, 34, 65, 56]
lst = [i*i for i in range(10)]
print(lst)
lsts = [i*i for i in range(10) if i%2 == 0]
print(lsts)