#IO files methos
#first one read
file = open('main.txt', 'r')#open is inbuild function in pythohn and {r} is read arrgument
print(file)
content = file.read() # {read()} is file content reader
print(content)
#_______________
#second method 
file1 = open('main1.txt', 'w')# this w is write mode on
contentWrite = file1.write("hello world") # write function is document text create help
print(contentWrite)
#third method same to write method append
file2 = open('main1.txt', 'a') #{a} is append this function means one file already create but document text added 
appendwrite = file2.write("hello wrold222")# this append function use {write} keyword 
print(appendwrite)