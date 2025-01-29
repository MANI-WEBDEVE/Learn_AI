import pyttsx3

def select_female_voice(voices):
    # Define keywords that might indicate a female voice
    female_keywords = ["female", "woman", "girl"]

    # Iterate through available voices and select the first one containing a female keyword
    for voice in voices:
        for keyword in female_keywords:
            if keyword in voice.name.lower():
                return voice

    # If no female voices found, return None
    return None

def say(text):
    try:
        # Initialize the pyttsx3 engine
        engine = pyttsx3.init()

        # Set properties (optional)
        engine.setProperty('rate', 150)  # Speed percent (can go over 100)
        engine.setProperty('volume', 0.9)  # Volume 0-1

        # Get available voices
        voices = engine.getProperty('voices')

        # Select a female voice (if available)
        female_voice = select_female_voice(voices)

        if female_voice:
            engine.setProperty('voice', female_voice.id)
        else:
            print("No female voices found. Using default voice.")

        # Say something
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("Error:", e)

say("Hello, I am a female voice.")
