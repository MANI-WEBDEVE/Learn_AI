import datetime

import speech_recognition as sr
import os
import subprocess
import pyttsx3
import subprocess
import sys
import datetime
# def say(text):
#     engine = pyttsx3.init()
#     engine.say(text)
#     engine.runAndWait()
##############################
import win32com.client
import speech_recognition as sr
import webbrowser
speaker = win32com.client.Dispatch("SAPI.SpVoice")

def say(text):
    speaker.Speak(text)

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.pause_threshold = 0.6
        print("Listening...")
        try:
            audio = r.listen(source)
            print("Audio captured")  # Add debug output
            query = r.recognize_google(audio, language="en-in")
            print(f"User said: {query}")
            return query
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that. Can you please repeat?")
            return ""
        except sr.RequestError as e:
            print(f"Request error: {e}")  # Add debug output
            print("Sorry, I couldn't request results at the moment. Please try again later.")
            return ""

say("   A I")
print("Listening...")
while True:
    query = takeCommand()
    sites =[["Youtube", "https://www.youtube.com"], ["10starhd", "https://10starhd.com"]]
    for site in sites:
        if f"Open {site[0]}".lower() in query.lower():
            say(f"Open in {site[0]} sir.....")
            webbrowser.open(site[1])

    if "open music" in query:
        musicPath = r"E:\python files\project\pythonProject1 JarvisAI\Heeriye (Official Video) Jasleen Royal ft Arijit Singh Dulquer Salmaan Aditya Sharma Taani Tanvir.mp3"  # Update with the correct path
        opener = "explorer" if sys.platform == "win32" else "xdg-open"  # Use "explorer" on Windows to open directories
        subprocess.call([opener, musicPath], shell=True)  # Adding shell=True for Windows
    if "the time" in query:
        print()
        Hour = datetime.datetime.now().strftime("%H:")
        Min = datetime.datetime.now().strftime("%M")
        say(f"sir time is {Hour} and {Min} minutes")

    if "Quran".lower() in query.lower():
        os.startfile(r"C:\Users\HP\Desktop\AlQuran")

    if "Open Editor".lower() in query.lower():
        os.startfile(r"C:\Users\HP\Desktop\Visual Studio Code")

    if "Terminal".lower() in query.lower():
        os.startfile(r"C:\Users\HP\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\System Tools\Command Prompt")