import os
#ya folder ko list karna ma madade kar ta ha
folders = os.listdir("pro")
#ya function {getcwd } directory dekhna ma madade karta ha
print(os.getcwd())
os.chdir("myenv")
#ya function {chdir} change directory karna ma madade karta ha
print(os.getcwd())
print(os.listdir(f"myenv"))
print(folders)
for i in folders:
    print(i)
    print(os.listdir(f"pro/{i}",))