import speech_recognition as sr

# Initialize the recognizer
recognizer = sr.Recognizer()

# Use the default microphone as the audio source
with sr.Microphone() as source:
    print("Please start speaking...")
    # Adjust for ambient noise
    recognizer.adjust_for_ambient_noise(source)
    # Listen for the user's input
    audio_data = recognizer.listen(source)

    try:
        print("Recognizing...")
        # Use the recognizer to convert speech to text
        text = recognizer.recognize_google(audio_data)
        print("You said:", text)
    except sr.UnknownValueError:
        print("Sorry, I could not understand what you said.")
    except sr.RequestError as e:
        print("Error occurred; {0}".format(e))
