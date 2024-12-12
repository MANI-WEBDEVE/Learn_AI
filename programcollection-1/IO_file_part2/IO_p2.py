file = open('myfile.txt', 'r')
i = 0
while True:
    i=  i+1
    lines = file.readline()
    if not lines:
      break
    m1 = int(lines.split(',')[0])
    m2 = int(lines.split(',')[1])
    m3 =int( lines.split(',')[2])
    print(f"the marks of student {i} in math {m1*2 }")
    print(f"the marks of student {i} in GK {m2*2}")
    print(f"the marks of student {i} in english {m3*2}")
    
print(lines)  
#first method for writelines method
file1 = open('myfile2.txt', 'w')
lines =['line1', 'line2', 'line3' ]
for line in lines:
   file1.write(line + '\n')
   print(type(lines))
#example no2 
file2 = open('myfile3.txt', 'w')
lines2 = ['linessss1\n', "linesss\n", 'linses3\n']
file2.writelines(lines2)