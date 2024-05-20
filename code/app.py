import random
import json
import torch
from google.cloud import speech_v1p1beta1 as speech  # Import the Google Speech-to-Text library

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Rest of your existing code remains unchanged

bot_name = "Sam"
print("Let's chat! (say 'quit' to exit)")

client = speech.SpeechClient()  # Initialize the Google Speech-to-Text client

while True:
    try:
        with sr.Microphone() as source:
            print("Speak:")
            audio = recognizer.listen(source)  # Capture audio input

        # Convert audio to text using Google Speech-to-Text API
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
        )
        audio_content = audio.frame_data
        audio = speech.RecognitionAudio(content=audio_content)

        response = client.recognize(config=config, audio=audio)
        sentence = response.results[0].alternatives[0].transcript

        if sentence.lower() == "quit":
            break

        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.8:
            for intent in intents['intents']:
                if tag == intent["intent"]:
                    if tag == "scheme_application":
                        additional_schemes = intent['additional_schemes']
                        print(f"{bot_name}: Here are some additional schemes:")
                        for scheme in additional_schemes:
                            
                            title = scheme['title']
                            how_to_avail = scheme['how_to_avail']
                            description = scheme['description']
                            print(f"Title: {title}\nHow to avail: {how_to_avail}\nDescription: {description}\n")
                    else:
                        print(f"{bot_name}: {(intent['responses'])}")
        else:
            print(f"{bot_name}: I do not understand...")

    except sr.UnknownValueError:
        print(f"{bot_name}: I couldn't understand what you said. Please try again.")
    except sr.RequestError:
        print(f"{bot_name}: There was an issue with the speech recognition service. Please check your internet connection.")

print("Goodbye!")
