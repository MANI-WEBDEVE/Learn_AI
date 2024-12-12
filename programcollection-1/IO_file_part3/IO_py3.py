#this is seek(), and tell() function
with open('file.txt','r')as f:
    print(type(f),f)
    f.seek(4)
    #read the next 5 bytes
    data = f.read()
    print(data)
#tell() function
with open('file.txt','r')as f:
    print(type(f),f)
    f.seek(4)
    #read the next 5 bytes
    print(f.tell())
    data = f.read()
    print(data)