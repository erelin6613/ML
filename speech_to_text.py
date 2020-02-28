import speech_recognition as sr
import pyaudio

r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something!")
    audio = r.record(source, duration=10)


WIT_AI_KEY = "RTGDXRZFO5OPVJRNSRB5BSDABGCFESDW"
try:
	with open('wit_token.txt', 'r') as file:
		print("You said: \n" + r.recognize_wit(audio, key=file.read().strip()))
except sr.UnknownValueError:
    print("Could not understand audio")
except sr.RequestError as e:
    print("Could not request results from service; {0}".format(e))