def arrange (fname, lname):
    print("heloo ", fname, lname);
arrange("inam", "tahir")
#Arbitaray function arguments
def collect (*name):
    print("Hello ", name[0], name[1],name[2]);
collect("inam", "tahir", "inam");
#keyword Arbitary function argumnets
def name(**name):
    print("hello",name["fname"],name["mname"],name["lname"])
name(mname = 'inam', lname = "khan", fname = "Muhammad")
#return statement
def avrage(*number):
    sum =0  
    for i in number:
        sum = sum +i
        return sum /len(number)
c = avrage(6,9,8,78)
print(c)