
#Al solution
# def ask_question(question, correct_answer):
#     user_answer = input(question + " ")

#     if user_answer.lower() == correct_answer.lower():
#         print("Correct!")
#         return True
#     else:
#         print("Incorrect. The correct answer is:", correct_answer)
#         return False

# def main():
#     score = 0

#     # Question 1
#     if ask_question("What is the capital of France?", "Paris"):
#         score += 1

#     # Question 2
#     if ask_question("What is 2 + 2?", "4"):
#         score += 1

#     # Add more questions as needed...

    # print("You scored {} out of {}.".format(score, 2))  # Adjust the total number of questions accordingly

# if __name__ == "__main__":
#     main()
#rewrite  this code again and add more question in this rewrite program
def askingque(ques, answere):
    userans = input(ques + "");
#-----------
#this if statement check corrct answere
    if userans.lower() == answere.lower():
        print("correct")
        return True
    else:
        print("you ansewere is wrong", answere)
        return False
    
#now write score and question function
def main():
    score = 0
    # question No 1
    if askingque("please tell me the capital of pakistan", "Islamabad"):
        score+= 1
    #question No2 
    if askingque("please tell me how to write first keyword function in python", "def"):
        score+=1
    #question No 3
    if askingque("the biggest mountain in pakistan ", "K2"):
        score+= 1
    #question No 4
    if askingque ("pleae write a release date of python ", "1989"):
        score+= 1
    #question NO 5
    if askingque ("please solve the addition question 4+900", "904"):
        score+= 1
    #question NO 6
    if askingque ("please solve the multiplication equation 5 * 100" ,"500"):
        score+= 1
    # print("your scored {} out of {}".format(score, 6))
    print(f"Your scored {score} out of {6}")
if __name__ == "__main__":
    main()