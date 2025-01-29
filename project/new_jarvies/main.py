import speech_recognition as sr
import pyaudio
import setuptools as stu
import pyttsx3
import webbrowser
import musicLibrary
import google.generativeai as genai
import requests
from gtts import gTTS
import pygame
import os
recognizer = sr.Recognizer()
# gemini_api_key = os.environ["gemini_api_key"]
# print(gemini_api_key)
engine = pyttsx3.init() 
apikey = ''
voices = engine.getProperty('voices')  

engine.setProperty('voice', voices[1].id)
def speak_old(text):
    engine.say(text)
    engine.runAndWait()

def speak(text):
    tts = gTTS(text)
    tts.save('temp.mp3')

    # Initialize pygame mixer
    pygame.mixer.init()
    try:
        # Load the MP3 file
        pygame.mixer.music.load('temp.mp3')
        print(f"Playing: {'temp.mp3'}")
        
        # Play the audio
        pygame.mixer.music.play()
        # Keep the program running while the audio plays
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    except pygame.error as e:
        print(f"Error playing audio: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        pygame.mixer.quit()
        os.remove('temp.mp3')
        # os.remove('temp.mp3')


def aiResponse(command):
    genai.configure(api_key="")
    generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 65536,
    "response_mime_type": "text/plain",
    }
    model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-thinking-exp-01-21",
    generation_config=generation_config,
    )


    chat_session = model.start_chat(
        history=[
            {
            "role": "user",
            "parts": [
                "kasa ho minglish ma response do",
            ],
            },
            {
            "role": "model",
            "parts": [
                "Kya bolti public?!  Sab badiya?  Main toh first class chillax kar raha hun.  Tu bata, scene kya hai?\n",
            ],
            },
            {
            "role": "user",
            "parts": [
                "hello",
            ],
            },
            {
            "role": "model",
            "parts": [
                "Hi there! How can I help you today?\n",
            ],
            },
            {
            "role": "user",
            "parts": [
                "please say hello in minglish language",
            ],
            },
            {
            "role": "model",
            "parts": [
                "Here's my thinking process to arrive at the Minglish \"Hello\" options:\n\n1. **Deconstruct the request:** The user wants \"hello\" in \"Minglish\".  This means a greeting that sounds like it's incorporating Hindi/Hinglish elements into English.  It needs to be casual and natural-sounding, reflecting how people actually speak.\n\n2. **Brainstorm core Minglish elements:** What are the common features of Minglish?\n    * **Hindi/Urdu words mixed in:**  \"Kya?\", \"Yaar\", \"Bhai\", \"Aur\", \"Thik hai\", \"Kaise ho\", \"Sab\", \"Kya haal hai\", \"Acha\", \"Chalo\", etc.\n    * **English words used with a Hindi/Indian inflection/pronunciation:**  \"Hello\" itself can be pronounced with a slightly different intonation.\n    * **Casual, informal tone:** Minglish is often used in informal settings, so the greetings should reflect that.\n    * **Code-switching:**  Switching between English and Hindi/Urdu within a sentence or phrase.\n\n3. **Generate initial ideas based on \"Hello\":**\n    * **Direct translation/Hindi equivalent:** \"Namaste\" (but this isn't Minglish, it's just Hindi).  \"Namaste\" *could* be used in a Minglish context, but it might be too formal for a simple \"hello\".\n    * **Simple English \"Hello\" with a Minglish flavor:**  How can we add flavor?\n        * **Exclamation/Enthusiasm:** \"Hello ji!\", \"Hello yaar!\", \"Hello bhai!\" (Adding Hindi kinship terms)\n        * **Questioning/Engaging:** \"Hello, kya haal hai?\", \"Hello, sab thik?\" (Adding Hindi phrases for \"how are you?\")\n        * **Casual English variations:** \"Hey!\", \"Hi!\" are already informal English greetings that fit the Minglish vibe.\n\n4. **Refine and categorize the options:**  Let's organize the brainstormed ideas into categories of Minglish greetings for \"hello\":\n\n    * **Simple & Direct:** These are closest to standard \"hello\" but with a Minglish touch.\n        * \"Hello ji!\" -  Polite and common, using \"ji\" for respect/politeness.\n        * \"Hello yaar!\" -  Friendly and casual, using \"yaar\" for \"friend\".\n        * \"Hello bhai!\" -  Brotherly/friendly, using \"bhai\" for \"brother/bro\".\n        * \"Hello hello!\" -  Playful and slightly informal, repetition for emphasis.\n        * \"Hey!\" -  Standard informal English, fits Minglish casualness.\n        * \"Hi!\" - Standard informal English, fits Minglish casualness.\n\n    * **Adding \"How are you?\" (Implied or Explicit):** Minglish greetings often combine \"hello\" with a quick \"how are you?\".\n        * \"Hello, kya haal hai?\" - \"Hello, what's the situation/how are you?\" (More formal \"haal chaal\" can also be used).\n        * \"Hello, sab thik?\" - \"Hello, is everything okay?\" (Casual and common).\n        * \"Hello, sab badiya?\" - \"Hello, is everything good?\" (Very casual and common).\n\n    * **More Expressive/Slangy (Optional - could be too much for just \"hello\"):**  While not strictly \"hello\" on its own, these are in the Minglish greeting realm and could be used in place of or alongside \"hello\" in a longer interaction.  (Maybe a bit too much for *just* \"hello\" but good to consider for the overall Minglish feel)\n        * \"Kya bolti public?\" - \"What do you say, people?\" (Very casual and slangy)\n        * \"Aur bhai?\" - \"And brother/bro?\" (Casual greeting, implying \"and how are you doing?\")\n\n5. **Select the best options and provide context:**  For a simple \"hello\" request, the simpler and more direct Minglish options are best.  I'll choose a few that represent different levels of casualness and politeness within Minglish.  It's important to explain *why* they are Minglish and what nuances they carry.  Hence, the explanation about adding \"ji,\" \"yaar,\" \"bhai,\" and the slightly more elaborate greetings with \"kya haal hai\" and \"sab thik\".\n\nThis thought process moves from understanding the request, brainstorming Minglish elements, generating options, refining them into categories, and finally selecting the most appropriate and representative examples with explanations.  It aims to be comprehensive yet focused on the specific request of \"hello in Minglish.\"",
                "Okay, here are a few ways to say \"hello\" in Minglish, ranging from very simple to a bit more flavorful:\n\n**Simple & Direct:**\n\n* **Hello ji!** (Adding \"ji\" makes it slightly more polite and Indian-sounding)\n* **Hello yaar!** (Adding \"yaar\" makes it friendly and casual, like \"hello friend\")\n* **Hello bhai!** (Adding \"bhai\" is like \"hello brother/bro\" - very informal and friendly)\n* **Hello hello!** (Just repeating \"hello\" can be a little playful and Minglish-y)\n* **Hey!** (Simple and informal English, perfectly fits in Minglish)\n* **Hi!** (Same as \"Hey\", informal and common in Minglish)\n\n**Slightly More Elaborate:**\n\n* **Hello, kya haal hai?** (Hello, how are you?)\n* **Hello, sab thik?** (Hello, is everything okay?)\n* **Hello, sab badiya?** (Hello, is everything good?)\n\n**Which one to use depends on the situation!**\n\n* For a very general, simple \"hello\",  \"**Hello ji!**\", \"**Hello yaar!**\", or just \"**Hey!**\" or \"**Hi!**\" are great.\n* If you want to be a bit more engaging, you can use  \"**Hello, kya haal hai?**\" or \"**Hello, sab thik?**\"\n\nLet me know if you want to try saying \"hello\" in a different way in Minglish! 😊",
            ],
            },
            {
            "role": "user",
            "parts": [
                "ya batao ka data science ka roadmap kiya hai",
            ],
            },
            {
            "role": "model",
            "parts": [
                "Here's a thinking process for generating a Data Science roadmap in Minglish:\n\n1. **Understand the request:** The user wants a \"roadmap for data science\" in \"Minglish\".  This means I need to outline the steps someone should take to learn data science, but explain it using a blend of Hindi/Urdu and English, in a casual, conversational style.\n\n2. **Identify key components of a Data Science roadmap:**  A typical roadmap includes:\n    * **Foundational skills:** Math, Statistics, Programming\n    * **Core Data Science skills:** Data manipulation, Machine Learning, Data Visualization,  Domain knowledge.\n    * **Tools and Technologies:** Programming languages (Python, R), Libraries (Pandas, NumPy, Scikit-learn), Databases, Cloud platforms (optional).\n    * **Learning Methodology:**  Online courses, projects, practice, community.\n    * **Career aspects:**  Building portfolio, job search, continuous learning.\n\n3. **Translate the components into Minglish concepts and phrasing:**  This is where the \"Minglish\" part comes in.  For each component, think of:\n    * **Hindi/Urdu equivalents or near-equivalents:**  e.g., \"Maths\" becomes \"maths/mathematics\", \"statistics\" becomes \"statistics/stats\", \"programming\" becomes \"programming/coding\".  \"Data manipulation\" could be \"data ko handle karna\", \"machine learning\" becomes \"machine learning/ML\".\n    * **Casual, conversational tone:** Use words like \"yaar\", \"bhai\", \"dekh\", \"sun\", \"basic\", \"advanced\", \"mast\", \"zabardast\", \"chill\".\n    * **Sentence structure and flow:** Mix English and Hindi/Urdu sentence structures naturally.  Don't just translate word-for-word; think about how someone would actually speak Minglish.\n    * **Use examples and analogies when possible:**  To make concepts clearer, use relatable examples.\n\n4. **Structure the roadmap logically:** Organize the components into a step-by-step guide. A good structure would be:\n    * **Introduction/Overview:** Briefly explain what data science is and why it's cool.\n    * **Step-by-Step Breakdown:**  List the steps in a logical order, from foundational to advanced.\n    * **Tips and Advice:** Include general guidance for learning.\n    * **Conclusion/Encouragement:**  End with a motivational message.\n\n5. **Draft the roadmap in Minglish, focusing on each step:**  Start writing, incorporating the Minglish elements identified in step 3.  For example:\n\n    * **Maths/Stats:** \"Dekh bhai, data science ka foundation maths aur stats hai.  Basic maths toh aana hi chahiye...\"\n    * **Programming:** \"Programming mein Python sabse mast hai...  Python seekhna bahut important hai...\"\n    * **Data Wrangling:** \"Data wrangling matlab data ko saaf suthra karna... Pandas library bahut kaam aati hai...\"\n    * **Machine Learning:** \"Machine learning is the heart of data science... Scikit-learn library bahut useful hai...\"\n    * **Data Visualization:** \"Data visualization matlab data ko picture form mein dikhana... matplotlib aur seaborn libraries use karna...\"\n    * **Projects/Practice:** \"Sirf padhne se kuch nahi hoga, projects banane padenge...  Real-world projects se hi experience milega...\"\n    * **Online Courses/Resources:** \"Online courses bahut available hain... Coursera, Udemy, DataCamp...  YouTube bhi dekho...\"\n\n6. **Review and Refine:**  Read through the drafted roadmap.  Check for:\n    * **Minglish fluency:** Does it sound natural? Is the Hindi/Urdu integrated well? Is the tone consistent?\n    * **Clarity and accuracy:** Is the information correct and easy to understand?  Are the steps logically ordered?\n    * **Completeness:**  Does it cover the key aspects of a data science roadmap?\n    * **Engagement:** Is it motivating and encouraging?\n\n7. **Add a concluding touch and call to action:**  End with a positive message and encourage the user to ask further questions.  \"Yeh toh basic roadmap hai.  Koi specific question hai toh pooch yaar!\"\n\nBy following these steps, I can create a comprehensive and engaging Data Science roadmap in Minglish that addresses the user's request effectively.  The key is to blend the technical information with a casual, relatable, and Minglish-flavored style.",
                "Haan yaar, definitely bataunga! Data science ka roadmap thoda lamba hai, lekin agar properly plan karega toh sab set ho jayega.  Suno dhyan se:\n\n**Data Science Roadmap - Minglish Style!**\n\n**Phase 1:  Basic Fundas Strong Karo (Building the Foundation)**\n\n1.  **Maths aur Stats ka Tadka (Math and Stats Flavor):**\n    *   **Basic Maths:**  Percentage, ratios, fractions, algebra - yeh sab toh aana hi chahiye.\n    *   **Statistics (Stats):**  Yeh bahut important hai!  Probability (chance kya hai?), averages (mean, median, mode), standard deviation (data kitna spread out hai), distributions (normal, binomial - yeh thoda advanced hai lekin pata hona chahiye).\n    *   **Linear Algebra:**  Matrices, vectors - yeh machine learning mein kaam aayega, thoda basic idea hona chahiye abhi.\n    *   **Calculus (Basic):** Derivatives, integrals -  yeh bhi machine learning ke liye important hai, lekin shuruwat mein thoda idea hi kaafi hai.\n\n    *   **Kaise Seekhe? (How to Learn?):**  Khan Academy is zabardast!  YouTube pe bahut saare free courses hain.  NCERT books (11th and 12th) bhi dekh sakta hai.\n\n2.  **Programming ka Hero Bano (Become a Programming Hero):**\n    *   **Python Sabse Best! (Python is the Best!):** Data science ke liye Python sabse popular hai. Easy to learn bhi hai aur bahut powerful bhi.\n    *   **Basics of Python:** Variables, data types (int, float, string), loops (for, while), conditions (if-else), functions, data structures (lists, dictionaries, tuples, sets).\n    *   **Libraries (Libraries - Toolkit):**\n        *   **NumPy:**  Maths aur numerical operations ke liye. Arrays ke saath khelne ke liye.\n        *   **Pandas:**  Data ko handle karne ke liye. Tables (DataFrames) bana sakta hai, data ko read kar sakta hai, clean kar sakta hai.  Yeh bahut important hai!\n\n    *   **Kaise Seekhe? (How to Learn?):** Codecademy, Coursera, Udemy, freeCodeCamp - bahut saare online platforms hain.  Practice karna bahut zaroori hai!  Code likh likh ke seekhega.\n\n**Phase 2: Data Science Skills ka Dhamaka (Data Science Skills Explosion)**\n\n3.  **Data Wrangling/Cleaning ka Kamaal (Data Wrangling/Cleaning Magic):**\n    *   **Real World Data Gandha Hota Hai! (Real-world data is messy!):** Data science mein 80% time data ko clean karne mein hi jata hai.\n    *   **Missing Values Handle Karna:**  Data mein agar kuch values missing hai toh usko kaise handle kare.\n    *   **Duplicate Data Hatana:**  Same data agar repeat ho raha hai toh usko remove karna.\n    *   **Data Transformation:**  Data ko format karna, scale karna, normalize karna - yeh sab important hai.\n\n    *   **Tools:** Pandas is king here!\n\n4.  **Data Analysis aur Visualization ka Jalwa (Data Analysis and Visualization Show):**\n    *   **Data ko Explore Karna:**  Data ko samajhna, patterns dhundhna, insights nikalna.\n    *   **Descriptive Statistics:** Mean, median, mode, standard deviation - yeh sab use karke data ko summarize karna.\n    *   **Data Visualization:**  Charts aur graphs bana ke data ko visually represent karna.  Bar charts, line charts, scatter plots, histograms, box plots - yeh sab banana seekhna.\n    *   **Libraries:**\n        *   **Matplotlib:** Basic plotting library.\n        *   **Seaborn:**  Advanced statistical plots banane ke liye.\n        *   **Plotly:** Interactive plots banane ke liye (optional, but cool!).\n\n5.  **Machine Learning ka Mantra (Machine Learning Mantra):**\n    *   **Machine Learning Kya Hai? (What is Machine Learning?):**  Computer ko data se seekhne dena, bina explicitly program kiye.\n    *   **Types of Machine Learning:**\n        *   **Supervised Learning:**  Labeled data se seekhna (classification, regression).\n        *   **Unsupervised Learning:** Unlabeled data se seekhna (clustering, dimensionality reduction).\n        *   **Reinforcement Learning:**  Environment se interact karke seekhna (games, robotics - thoda advanced).\n    *   **Algorithms (Algorithms - Recipes):**\n        *   **Linear Regression:**  Simple regression problem ke liye.\n        *   **Logistic Regression:**  Classification problem ke liye.\n        *   **Decision Trees:**  Classification aur regression dono ke liye.\n        *   **Random Forests:**  Decision trees ka improved version.\n        *   **Support Vector Machines (SVM):**  Powerful classification algorithm.\n        *   **K-Means Clustering:**  Clustering ke liye.\n    *   **Model Evaluation:**  Model kitna accha perform kar raha hai, yeh kaise check kare (accuracy, precision, recall, F1-score, etc.).\n    *   **Libraries:**\n        *   **Scikit-learn (sklearn):**  Yeh sabse important library hai machine learning ke liye.  Algorithms, model evaluation tools sab kuch isme hai.\n\n**Phase 3:  Advanced Level aur Projects ka Tadka (Advanced Level and Projects Flavor)**\n\n6.  **Deep Learning ka Dive (Deep Learning Dive - Optional but Hot):**\n    *   **Neural Networks:**  Brain jaise kaam karne wale networks.\n    *   **Deep Learning Frameworks:**\n        *   **TensorFlow:** Google ka framework.\n        *   **Keras:**  TensorFlow ke upar easy-to-use interface.\n        *   **PyTorch:**  Facebook ka framework, research mein popular.\n\n    *   **Kaise Seekhe? (How to Learn?):**  Fast.ai, deeplearning.ai (Coursera) - yeh sab bahut acche resources hain.\n\n7.  **Domain Knowledge ka Importance (Importance of Domain Knowledge):**\n    *   **Kis Field Mein Data Science Apply Karna Hai? (Which field you want to apply data science to?):**  Finance, healthcare, marketing, e-commerce - koi bhi field choose kar sakta hai.\n    *   **Domain Knowledge Se Problem Samajhne Mein Help Milegi (Domain knowledge will help in understanding the problem):**  Agar finance mein data science kar raha hai toh finance ke basics pata hone chahiye.\n\n8.  **Projects aur Portfolio Banao (Build Projects and Portfolio):**\n    *   **Sirf Padhaai Se Kuch Nahi Hoga! (Just studying won't be enough!):** Real-world projects karna bahut zaroori hai.\n    *   **Kaggle:**  Data science competitions ke liye best platform. Datasets bhi mil jayenge practice ke liye.\n    *   **GitHub:**  Projects ko GitHub pe upload karo. Yeh portfolio ban jayega.\n\n**Phase 4:  Job Search aur Continuous Learning ka Cycle (Job Search and Continuous Learning Cycle)**\n\n9.  **Resume aur Interview Prep (Resume and Interview Preparation):**\n    *   **Resume Strong Banao:** Projects, skills, experience sab highlight karo.\n    *   **Interview Questions Practice Karo:**  Technical questions (maths, stats, programming, machine learning) aur behavioral questions sab practice karo.\n    *   **LeetCode, HackerRank:** Coding interview practice ke liye.\n\n10. **Continuous Learning - Yeh Kabhi Khatam Nahi Hota! (Continuous Learning - It Never Ends!):**\n    *   **Data science field bahut fast change hota hai.**  Naye algorithms, naye tools aate rehte hain.\n    *   **Online Courses, Blogs, Research Papers padhte raho.**  Keep learning!\n\n**Important Tips - Yaad Rakhna! (Important Tips - Remember!)**\n\n*   **Practice, Practice, Practice!**  Coding aur data analysis jitna practice karega, utna hi better hoga.\n*   **Patience Rakhna! (Be Patient!):** Data science time leta hai seekhne mein.  Frustrate mat hona.\n*   **Community Join Karo! (Join the Community!):** Online forums, meetups join karo.  Dusron se seekhne ko milega.\n*   **Questions Poocho! (Ask Questions!):**  Agar kuch samajh nahi aa raha hai toh poochne mein sharmao mat.\n\nYeh toh ek basic roadmap hai.  Specific interest aur time ke hisab se isko adjust kar sakta hai.  Lekin yeh sab steps follow karega toh data science mein definitely aage badhega!\n\nKoi aur question hai toh pooch yaar! All the best! 👍",
            ],
            },
        ]
)

    response = chat_session.send_message("You are a virtual assitent like alexa and response only minglish language and add the emojies in the response as well and short ma jawad da zada sa zada tow ya three line only")
    response = chat_session.send_message(command)

    return (response.text)

def processCommand(command):
    if ("open google" in command.lower()):
        webbrowser.open("https://www.google.com")
    elif (command.lower().startswith("play")):
        print("neter")
        music = command.lower().split(" ")[1]
        print(music)
        song = musicLibrary.music[music]
        print(song)
        webbrowser.open(song)
    elif ("news" == command.lower()):
        req = requests.get(f"https://newsapi.org/v2/top-headlines?country=us&apiKey={apikey}")

        result = req.json()
        articles = result['articles']
        for article in articles:
            speak(article['title'])
    else:
        response_text = aiResponse(command)
        speak(response_text)
        print("user:", command)
        print("AI:", response_text)

if __name__ == "__main__":
    speak("Intializing bolt AI")
    while True:
        r = sr.Recognizer()
        r.pause_threshold = 1  # Add this line to adjust sensitivity
        
        try:
            with sr.Microphone() as source:
                print("Listening🤖")
                # Adjust energy threshold for ambient noise
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio1 = r.listen(source, phrase_time_limit=1)  # Removed timeout, increased phrase_time_limit
            
            try:
                wake_word = r.recognize_google(audio1, language="en-in").lower()
                print(f"Heard: {wake_word}")
                    
                    # ===== Improved Activation =====
                if "gemini" in wake_word:
                    print("activate AI")
                    speak("yes sir")
                    print("yes sir")
                    with sr.Microphone() as source:
                        print("Listening...")
                        r.adjust_for_ambient_noise(source, duration=0.5)
                        audio = r.listen(source, phrase_time_limit=5)
                        command = r.recognize_google(audio)
                        processCommand(command)
            
            except sr.UnknownValueError:
                print("Sorry, I didn't catch that. Please try again.")
            except sr.RequestError:
                print("Sorry, there was an error with the speech recognition service")
                
        except sr.WaitTimeoutError:
            print("No speech detected. Please try again.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

