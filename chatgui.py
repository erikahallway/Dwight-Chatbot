
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import json
import random
intents = json.loads(open('intents.json').read())
# use natural language toolkit
import nltk
from nltk.stem.lancaster import LancasterStemmer
# word stemmer
stemmer = LancasterStemmer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
# 3 classes of training data
training_data = []
training_data.append({"class":"greeting", "sentence":"Hi"})
training_data.append({"class":"greeting", "sentence":"Hello"})
training_data.append({"class":"greeting", "sentence":"Hey man"})
training_data.append({"class":"greeting", "sentence":"Hey"})
training_data.append({"class":"greeting", "sentence":"how are you?"})
training_data.append({"class":"greeting", "sentence":"How’s it going"})
training_data.append({"class":"greeting", "sentence":"How do you do"})
training_data.append({"class":"greeting", "sentence":"How are you doing"})
training_data.append({"class":"greeting", "sentence":"How’s everything"})
training_data.append({"class":"greeting", "sentence":"How are things"})
training_data.append({"class":"greeting", "sentence":"How’s life"})
training_data.append({"class":"greeting", "sentence":"How’s your day"})
training_data.append({"class":"greeting", "sentence":"how is your day?"})
training_data.append({"class":"greeting", "sentence":"good day"})
training_data.append({"class":"greeting", "sentence":"how is it going today?"})
training_data.append({"class":"greeting", "sentence":"What is up?"})
training_data.append({"class":"greeting", "sentence":"How’s your day going"})
training_data.append({"class":"greeting", "sentence":"Good to see you"})
training_data.append({"class":"greeting", "sentence":"Nice to meet you"})
training_data.append({"class":"greeting", "sentence":"What’s up"})
training_data.append({"class":"greeting", "sentence":"What’s new"})
training_data.append({"class":"greeting", "sentence":"What’s going on"})
training_data.append({"class":"greeting", "sentence":"Good morning"})
training_data.append({"class":"greeting", "sentence":"Good afternoon"})
training_data.append({"class":"greeting", "sentence":"Good evening"})

training_data.append({"class":"goodbye", "sentence":"have a nice day"})
training_data.append({"class":"goodbye", "sentence":"see you later"})
training_data.append({"class":"goodbye", "sentence":"have a nice day"})
training_data.append({"class":"goodbye", "sentence":"talk to you soon"})
training_data.append({"class":"goodbye", "sentence":"Goodbye"})
training_data.append({"class":"goodbye", "sentence":"Bye"})
training_data.append({"class":"goodbye", "sentence":"Bye bye"})
training_data.append({"class":"goodbye", "sentence":"See you later"})
training_data.append({"class":"goodbye", "sentence":"See you soon"})
training_data.append({"class":"goodbye", "sentence":"Talk to you later"})
training_data.append({"class":"goodbye", "sentence":"I must be going"})
training_data.append({"class":"goodbye", "sentence":"Take it easy"})
training_data.append({"class":"goodbye", "sentence":"I’m off"})
training_data.append({"class":"goodbye", "sentence":"Have a nice day"})
training_data.append({"class":"goodbye", "sentence":"Have a good weekend"})
training_data.append({"class":"goodbye", "sentence":"Have a good vacation"})
training_data.append({"class":"goodbye", "sentence":"Have a good evening"})
training_data.append({"class":"goodbye", "sentence":"It was nice to see you"})
training_data.append({"class":"goodbye", "sentence":"Goodnight"})

training_data.append({"class":"sandwich", "sentence":"make me a sandwich"})
training_data.append({"class":"sandwich", "sentence":"can you make a sandwich?"})
training_data.append({"class":"sandwich", "sentence":"having a sandwich today?"})
training_data.append({"class":"sandwich", "sentence":"what's for lunch?"})

training_data.append({"class":"identify", "sentence":"Dwight?"})

training_data.append({"class":"thanks", "sentence":"Thank you"})
training_data.append({"class":"thanks", "sentence":"Thanks"})
training_data.append({"class":"thanks", "sentence":"Thanks so much"})
training_data.append({"class":"thanks", "sentence":"That’s very kind of you"})
training_data.append({"class":"thanks", "sentence":"You made my day"})
training_data.append({"class":"thanks", "sentence":"You’re awesome"})
training_data.append({"class":"thanks", "sentence":"I couldn’t have done it without you"})
training_data.append({"class":"thanks", "sentence":"I really want to thank you for your help"})
training_data.append({"class":"thanks", "sentence":"I really appreciate everything you’ve done"})
training_data.append({"class":"thanks", "sentence":"I’m grateful for your help"})
training_data.append({"class":"thanks", "sentence":"This means a lot to me"})
training_data.append({"class":"thanks", "sentence":"Thanks for having my back"})
training_data.append({"class":"thanks", "sentence":"I owe you one"})
training_data.append({"class":"thanks", "sentence":"Thank you for your guidance"})
training_data.append({"class":"thanks", "sentence":"I appreciate your feedback"})
training_data.append({"class":"thanks", "sentence":"I’m grateful for your assistance"})

training_data.append({"class":"thanks_for_sharing", "sentence":"Thank you for your sharing"})
training_data.append({"class":"thanks_for_sharing", "sentence":"Thank you for sharing"})
training_data.append({"class":"thanks_for_sharing", "sentence":"Thanks for your sharing"})
training_data.append({"class":"thanks_for_sharing", "sentence":"Thanks for sharing"})
training_data.append({"class":"thanks_for_sharing", "sentence":"Thank you for telling me"})
training_data.append({"class":"thanks_for_sharing", "sentence":"Thanks for telling me"})

training_data.append({"class":"positive_answer_request", "sentence":"Can I ask you a question?"})
training_data.append({"class":"positive_answer_request", "sentence":"Can I ask you something?"})
training_data.append({"class":"positive_answer_request", "sentence":"Can you tell me something? "})
training_data.append({"class":"positive_answer_request", "sentence":"I need you to tell me something."})
training_data.append({"class":"positive_answer_request", "sentence":"Can you do something for me?"})

training_data.append({"class":"positive_answer_agreement", "sentence":"Don’t you agree?"})
training_data.append({"class":"positive_answer_agreement", "sentence":"Don’t you agree with me?"})
training_data.append({"class":"positive_answer_agreement", "sentence":"Do you agree?"})
training_data.append({"class":"positive_answer_agreement", "sentence":"Do you agree with me?"})

training_data.append({"class":"positive_answer_confirmation_speaker", "sentence":"Did I already ask you about this?"})
training_data.append({"class":"positive_answer_confirmation_speaker", "sentence":"Did I ask you about this?"})
training_data.append({"class":"positive_answer_confirmation_speaker", "sentence":"Did I already ask about this?"})
training_data.append({"class":"positive_answer_confirmation_speaker", "sentence":"Did I ask about this?"})

training_data.append({"class":"really", "sentence":"Oh, really?"})
training_data.append({"class":"really", "sentence":"Really?"})
training_data.append({"class":"really", "sentence":"Is that true?"})

training_data.append({"class":"happy", "sentence":"Yay!"})
training_data.append({"class":"happy", "sentence":"I’m soooo happy!"})
training_data.append({"class":"happy", "sentence":"I’m happy!"})
training_data.append({"class":"happy", "sentence":"Sooooo happy!"})

training_data.append({"class":"negative_response", "sentence":"It was terrible."})
training_data.append({"class":"negative_response", "sentence":"It’s terrible."})
training_data.append({"class":"negative_response", "sentence":"It is terrible."})
training_data.append({"class":"negative_response", "sentence":"It was awful."})
training_data.append({"class":"negative_response", "sentence":"It’s awful."})
training_data.append({"class":"negative_response", "sentence":"It is awful."})

training_data.append({"class":"negative_answer", "sentence":"What should I do?"})
training_data.append({"class":"negative_answer", "sentence":"I don’t know what to do."})
training_data.append({"class":"negative_answer", "sentence":"What would you do?"})
training_data.append({"class":"negative_answer", "sentence":"I’m not sure what to do."})
training_data.append({"class":"negative_answer", "sentence":"What’s your opinion?"})
training_data.append({"class":"negative_answer", "sentence":"I need advice."})

training_data.append({"class":"options", "sentence":"What do you want?"})
training_data.append({"class":"options", "sentence":"What do you think?"})

training_data.append({"class":"name", "sentence":"Who are you?"})
training_data.append({"class":"name", "sentence":"What’s your name?"})
training_data.append({"class":"name", "sentence":"What is your name?"})
training_data.append({"class":"name", "sentence":"Your name?"})
training_data.append({"class":"name", "sentence":"Could you please tell me your name?"})
training_data.append({"class":"name", "sentence":"Who am I speaking with?"})
training_data.append({"class":"name", "sentence":"Can I get your name?"})
training_data.append({"class":"name", "sentence":"What is your full name?"})
training_data.append({"class":"name", "sentence":"I’m sorry, I forgot your name."})

training_data.append({"class":"myself", "sentence":"Talk about your experience."})
training_data.append({"class":"myself", "sentence":"Tell me about your experience."})
training_data.append({"class":"myself", "sentence":"Tell me about yourself."})
training_data.append({"class":"myself", "sentence":"Share me with your experience."})
training_data.append({"class":"myself", "sentence":"Could you please tell me more about yourself?"})
training_data.append({"class":"myself", "sentence":"Could you tell me more about yourself?"})
training_data.append({"class":"myself", "sentence":"How’s your experience?"})
training_data.append({"class":"myself", "sentence":"What’s your experience?"})
training_data.append({"class":"myself", "sentence":"I want to know more about you."})

training_data.append({"class":"story", "sentence":"Tell me a funny story."})
training_data.append({"class":"story", "sentence":"Tell me a story."})
training_data.append({"class":"story", "sentence":"Could you tell me a story?"})

training_data.append({"class":"martial_Arts", "sentence":"What do you think about Martial Arts?"})
training_data.append({"class":"martial_Arts", "sentence":"What do you think about Karate?"})
training_data.append({"class":"martial_Arts", "sentence":"What do you think about Tae Kwon Do?"})
training_data.append({"class":"martial_Arts", "sentence":"What do you think about Jiu Jitsu?"})
training_data.append({"class":"martial_Arts", "sentence":"What do you think about Ninjas?"})
training_data.append({"class":"martial_Arts", "sentence":"What do you think about martial arts?"})
training_data.append({"class":"martial_Arts", "sentence":"What do you think about karate?"})
training_data.append({"class":"martial_Arts", "sentence":"What do you think about tae kwon do?"})
training_data.append({"class":"martial_Arts", "sentence":"What do you think about jiu jitsu?"})
training_data.append({"class":"martial_Arts", "sentence":"What do you think about ninjas?"})

training_data.append({"class":"career_suggestion", "sentence":"Could you please give me some advice?"})
training_data.append({"class":"career_suggestion", "sentence":"Could you please give me some career advice?"})
training_data.append({"class":"career_suggestion", "sentence":"Do you have any advice for my career?"})
training_data.append({"class":"career_suggestion", "sentence":"Could you please tell me your career experience?"})
training_data.append({"class":"career_suggestion", "sentence":"Could you give me some advice?"})
training_data.append({"class":"career_suggestion", "sentence":"Could you give me some career advice?"})
training_data.append({"class":"career_suggestion", "sentence":"Do you have any career advice?"})
training_data.append({"class":"career_suggestion", "sentence":"Could you tell me your career experience?"})
training_data.append({"class":"career_suggestion", "sentence":"What’s your career experience?"})

training_data.append({"class":"relationship", "sentence":"What’s your opinion on a relationship?"})
training_data.append({"class":"relationship", "sentence":"relationship"})
training_data.append({"class":"relationship", "sentence":"Could you please give me some advice on a relationship?"})

training_data.append({"class":"suggestion", "sentence":"What’s your suggestion?"})
training_data.append({"class":"suggestion", "sentence":"Any suggestion?"})
training_data.append({"class":"suggestion", "sentence":"What’s your opinion?"})

training_data.append({"class":"what_are_you_doing", "sentence":"What are you doing?"})
training_data.append({"class":"what_are_you_doing", "sentence":"Whatcha doing?"})
training_data.append({"class":"what_are_you_doing", "sentence":"What are you up to?"})
training_data.append({"class":"what_are_you_doing", "sentence":"Are you busy?"})
training_data.append({"class":"what_are_you_doing", "sentence":"Why are you busy?"})

training_data.append({"class":"argument_suggestion", "sentence":"what do you think about sports?"})
training_data.append({"class":"argument_suggestion", "sentence":"what's your favorite sport?"})
training_data.append({"class":"argument_suggestion", "sentence":"what are your favorite sports?"})
training_data.append({"class":"argument_suggestion", "sentence":"what do you think about women's sports?"})

training_data.append({"class":"health_suggestion", "sentence":"what about health?"})
training_data.append({"class":"health_suggestion", "sentence":"how about health?"})
training_data.append({"class":"health_suggestion", "sentence":"tell me about health"})
training_data.append({"class":"health_suggestion", "sentence":"what do you think about health?"})
training_data.append({"class":"health_suggestion", "sentence":"what about healthcare?"})
training_data.append({"class":"health_suggestion", "sentence":"how about healthcare?"})
training_data.append({"class":"health_suggestion", "sentence":"tell me about healthcare"})
training_data.append({"class":"health_suggestion", "sentence":"what do you think about healthcare?"})
training_data.append({"class":"health_suggestion", "sentence":"should I go to the doctor?"})
training_data.append({"class":"health_suggestion", "sentence":"should I go to the hospital?"})
training_data.append({"class":"health_suggestion", "sentence":"should I see the doctor?"})
training_data.append({"class":"health_suggestion", "sentence":"should I go see the doctor?"})

training_data.append({"class":"wisdom", "sentence":"could I get some advice?"})
training_data.append({"class":"wisdom", "sentence":"can I get some advice?"})
training_data.append({"class":"wisdom", "sentence":"could I get advice?"})
training_data.append({"class":"wisdom", "sentence":"can I get advice?"})
training_data.append({"class":"wisdom", "sentence":"I need some advice?"})
training_data.append({"class":"wisdom", "sentence":"I need advice?"})
training_data.append({"class":"wisdom", "sentence":"what's your advice?"})
training_data.append({"class":"wisdom", "sentence":"what is your advice?"})
training_data.append({"class":"wisdom", "sentence":"what's your advice to me?"})
training_data.append({"class":"wisdom", "sentence":"what is your advice to me?"})
training_data.append({"class":"wisdom", "sentence":"what's your advice for me?"})
training_data.append({"class":"wisdom", "sentence":"what is your advice for me?"})

training_data.append({"class":"assumption", "sentence":"I have an assumption"})
training_data.append({"class":"assumption", "sentence":"it's an assumption"})
training_data.append({"class":"assumption", "sentence":"it is an assumption"})
training_data.append({"class":"assumption", "sentence":"do you think it's possible?"})
training_data.append({"class":"assumption", "sentence":"do you think it's possible for me?"})
training_data.append({"class":"assumption", "sentence":"do you think it is possible?"})
training_data.append({"class":"assumption", "sentence":"do you think it is possible for me?"})
training_data.append({"class":"assumption", "sentence":"is it possible?"})
training_data.append({"class":"assumption", "sentence":"it's possible"})
training_data.append({"class":"assumption", "sentence":"it is possible"})

training_data.append({"class":"complain_to_me", "sentence":"could you be nice?"})
training_data.append({"class":"complain_to_me", "sentence":"could you be nicer?"})
training_data.append({"class":"complain_to_me", "sentence":"could you be kind?"})
training_data.append({"class":"complain_to_me", "sentence":"could you be friendly?"})
training_data.append({"class":"complain_to_me", "sentence":"can you be nice?"})
training_data.append({"class":"complain_to_me", "sentence":"can you be nicer?"})
training_data.append({"class":"complain_to_me", "sentence":"can you be kind?"})
training_data.append({"class":"complain_to_me", "sentence":"can you be friendly?"})
training_data.append({"class":"complain_to_me", "sentence":"why are you rude?"})
training_data.append({"class":"complain_to_me", "sentence":"why are you being mean?"})
training_data.append({"class":"complain_to_me", "sentence":"why are you so rude?"})
training_data.append({"class":"complain_to_me", "sentence":"why are you being so mean?"})
training_data.append({"class":"complain_to_me", "sentence":"are you angry?"})
training_data.append({"class":"complain_to_me", "sentence":"are you upset?"})
training_data.append({"class":"complain_to_me", "sentence":"are you mad?"})

training_data.append({"class":"ask_to_do", "sentence":"could you help me with it?"})
training_data.append({"class":"ask_to_do", "sentence":"could you help me with something?"})
training_data.append({"class":"ask_to_do", "sentence":"could you help me?"})
training_data.append({"class":"ask_to_do", "sentence":"could you help me out?"})
training_data.append({"class":"ask_to_do", "sentence":"could you give me some help?"})
training_data.append({"class":"ask_to_do", "sentence":"can you help me with it?"})
training_data.append({"class":"ask_to_do", "sentence":"can you help me with something?"})
training_data.append({"class":"ask_to_do", "sentence":"can you help me?"})
training_data.append({"class":"ask_to_do", "sentence":"can you help me out?"})
training_data.append({"class":"ask_to_do", "sentence":"can you give me some help?"})
training_data.append({"class":"ask_to_do", "sentence":"I need your help"})
training_data.append({"class":"ask_to_do", "sentence":"please help me"})
training_data.append({"class":"ask_to_do", "sentence":"I need you to help me"})

training_data.append({"class":"decision", "sentence":"is that really your decision?"})
training_data.append({"class":"decision", "sentence":"is that your decision?"})

training_data.append({"class":"age_concern", "sentence":"I’m worried about my age"})
training_data.append({"class":"age_concern", "sentence":"I’m anxious about my age"})
training_data.append({"class":"age_concern", "sentence":"I’m worried about how old I am"})
training_data.append({"class":"age_concern", "sentence":"I’m anxious about how old I am"})
training_data.append({"class":"age_concern", "sentence":"I’m stressed about how old I am"})
training_data.append({"class":"age_concern", "sentence":"I’m too old"})
training_data.append({"class":"age_concern", "sentence":"I’m too young"})
training_data.append({"class":"age_concern", "sentence":"I am worried about my age"})
training_data.append({"class":"age_concern", "sentence":"I am anxious about my age"})
training_data.append({"class":"age_concern", "sentence":"I am worried about how old I am"})
training_data.append({"class":"age_concern", "sentence":"I am anxious about how old I am"})
training_data.append({"class":"age_concern", "sentence":"I am stressed about how old I am"})
training_data.append({"class":"age_concern", "sentence":"I am too old"})
training_data.append({"class":"age_concern", "sentence":"I am too young"})
training_data.append({"class":"age_concern", "sentence":"why am I so old?"})
training_data.append({"class":"age_concern", "sentence":"why am I so young?"})

training_data.append({"class":"ideal_valentine_day", "sentence":"what’s your perfect valentine's day?"})
training_data.append({"class":"ideal_valentine_day", "sentence":"what’s your ideal valentine's day?"})
training_data.append({"class":"ideal_valentine_day", "sentence":"what is your perfect valentine's day?"})
training_data.append({"class":"ideal_valentine_day", "sentence":"what is your ideal valentine's day?"})
training_data.append({"class":"ideal_valentine_day", "sentence":"what should I do for valentine's day?"})

training_data.append({"class":"fun_fact", "sentence":"tell me some funny things"})
training_data.append({"class":"fun_fact", "sentence":"tell me something funny"})

training_data.append({"class":"joke", "sentence":"tell me a joke"})
training_data.append({"class":"joke", "sentence":"tell a joke"})
training_data.append({"class":"joke", "sentence":"tell a joke to me"})
training_data.append({"class":"joke", "sentence":"could you tell me a joke?"})
training_data.append({"class":"joke", "sentence":"could you tell a joke?"})
training_data.append({"class":"joke", "sentence":"could you tell a joke to me?"})
training_data.append({"class":"joke", "sentence":"could you tell a joke for me?"})
training_data.append({"class":"joke", "sentence":"can you tell me a joke?"})
training_data.append({"class":"joke", "sentence":"can you tell a joke?"})
training_data.append({"class":"joke", "sentence":"can you tell a joke to me?"})
training_data.append({"class":"joke", "sentence":"can you tell a joke for me?"})
training_data.append({"class":"joke", "sentence":"do you have any other jokes?"})
training_data.append({"class":"joke", "sentence":"do you know any other jokes?"})

training_data.append({"class":"life_is_tough", "sentence":"life is hard"})
training_data.append({"class":"life_is_tough", "sentence":"life is so hard"})
training_data.append({"class":"life_is_tough", "sentence":"life is hard for me"})
training_data.append({"class":"life_is_tough", "sentence":"life is tough"})
training_data.append({"class":"life_is_tough", "sentence":"life is so tough"})
training_data.append({"class":"life_is_tough", "sentence":"life is tough for me"})
training_data.append({"class":"life_is_tough", "sentence":"everything sucks"})
training_data.append({"class":"life_is_tough", "sentence":"everything's falling apart"})
training_data.append({"class":"life_is_tough", "sentence":"everything is falling apart"})
training_data.append({"class":"life_is_tough", "sentence":"I had a bad day"})

training_data.append({"class":"crazy_love", "sentence":"I need love"})
training_data.append({"class":"crazy_love", "sentence":"I cannot live without love"})
training_data.append({"class":"crazy_love", "sentence":"I can't live without love"})
training_data.append({"class":"crazy_love", "sentence":"I'm lonely"})
training_data.append({"class":"crazy_love", "sentence":"I am lonely"})
training_data.append({"class":"crazy_love", "sentence":"I'm so lonely"})
training_data.append({"class":"crazy_love", "sentence":"I am so lonely"})
training_data.append({"class":"crazy_love", "sentence":"I want a boyfriend"})
training_data.append({"class":"crazy_love", "sentence":"I want a girlfriend"})
training_data.append({"class":"crazy_love", "sentence":"I need a boyfriend"})
training_data.append({"class":"crazy_love", "sentence":"I need a girlfriend"})
training_data.append({"class":"crazy_love", "sentence":"you need a girlfriend"})
training_data.append({"class":"crazy_love", "sentence":"why don’t I have a boyfriend?"})
training_data.append({"class":"crazy_love", "sentence":"why don’t I have a boyfriend?"})

training_data.append({"class":"power_point", "sentence":"what do you think about powerpoint?"})
training_data.append({"class":"power_point", "sentence":"what about powerpoint?"})
training_data.append({"class":"power_point", "sentence":"how about powerpoint?"})
training_data.append({"class":"power_point", "sentence":"what do you think about presentations?"})
training_data.append({"class":"power_point", "sentence":"what about presentations?"})
training_data.append({"class":"power_point", "sentence":"how about presentations?"})
training_data.append({"class":"power_point", "sentence":"I hate presentations"})
training_data.append({"class":"power_point", "sentence":"I don't like presentations"})
training_data.append({"class":"power_point", "sentence":"I like presentations"})
training_data.append({"class":"power_point", "sentence":"I love presentations"})
training_data.append({"class":"power_point", "sentence":"I'm scared of presentations"})
training_data.append({"class":"power_point", "sentence":"I hate public speaking"})
training_data.append({"class":"power_point", "sentence":"I don't like public speaking"})
training_data.append({"class":"power_point", "sentence":"I like public speaking"})
training_data.append({"class":"power_point", "sentence":"I love public speaking"})
training_data.append({"class":"power_point", "sentence":"I'm scared of public speaking"})

training_data.append({"class":"smoke", "sentence":"what do you think about smoking?"})
training_data.append({"class":"smoke", "sentence":"what about smoking?"})
training_data.append({"class":"smoke", "sentence":"how about smoking?"})
training_data.append({"class":"smoke", "sentence":"what's your opinion about smoking?"})
training_data.append({"class":"smoke", "sentence":"what is your opinion about smoking?"})
training_data.append({"class":"smoke", "sentence":"I love smoking"})
training_data.append({"class":"smoke", "sentence":"I hate smoking"})
training_data.append({"class":"smoke", "sentence":"smoking is gross"})

training_data.append({"class":"star_wars", "sentence":"what do you think about star wars?"})
training_data.append({"class":"star_wars", "sentence":"what about star wars?"})
training_data.append({"class":"star_wars", "sentence":"how about star wars?"})
training_data.append({"class":"star_wars", "sentence":"would you be a padawan or a jedi?"})
training_data.append({"class":"star_wars", "sentence":"who's fault is it?"})
training_data.append({"class":"star_wars", "sentence":"who would you be in star wars?"})

training_data.append({"class":"gun_issue", "sentence":"what do you think about guns?"})
training_data.append({"class":"gun_issue", "sentence":"what about guns?"})
training_data.append({"class":"gun_issue", "sentence":"how about guns?"})
training_data.append({"class":"gun_issue", "sentence":"what do you think about gun issues?"})
training_data.append({"class":"gun_issue", "sentence":"what about gun issues?"})
training_data.append({"class":"gun_issue", "sentence":"how about gun issues?"})
training_data.append({"class":"gun_issue", "sentence":"what do you think about the 2nd amendment?"})
training_data.append({"class":"gun_issue", "sentence":"what about the 2nd amendment?"})
training_data.append({"class":"gun_issue", "sentence":"how about the 2nd amendment?"})
training_data.append({"class":"gun_issue", "sentence":"what do you think about the second amendment?"})
training_data.append({"class":"gun_issue", "sentence":"what about the second amendment?"})
training_data.append({"class":"gun_issue", "sentence":"how about the second amendment?"})
training_data.append({"class":"gun_issue", "sentence":"what do you think about gun laws?"})
training_data.append({"class":"gun_issue", "sentence":"what about gun laws?"})
training_data.append({"class":"gun_issue", "sentence":"how about gun laws?"})
training_data.append({"class":"gun_issue", "sentence":"what do you think security?"})
training_data.append({"class":"gun_issue", "sentence":"what about security?"})
training_data.append({"class":"gun_issue", "sentence":"how about security?"})

training_data.append({"class":"triceratops", "sentence":"what do you think about dinosaurs?"})
training_data.append({"class":"triceratops", "sentence":"what about dinosaurs?"})
training_data.append({"class":"triceratops", "sentence":"what about the dinosaurs?"})
training_data.append({"class":"triceratops", "sentence":"how about dinosaurs?"})
training_data.append({"class":"triceratops", "sentence":"how about the dinosaurs?"})
training_data.append({"class":"triceratops", "sentence":"what's the coolest dinosaur?"})
training_data.append({"class":"triceratops", "sentence":"what is the coolest dinosaur?"})
training_data.append({"class":"triceratops", "sentence":"tell me about dinosaurs"})
training_data.append({"class":"triceratops", "sentence":"what do you think about dinosaurs?"})

training_data.append({"class":"downsize", "sentence":"should we downsize?"})
training_data.append({"class":"downsize", "sentence":"should I downsize?"})
training_data.append({"class":"downsize", "sentence":"should my company downsize?"})
training_data.append({"class":"downsize", "sentence":"should my branch downsize?"})
training_data.append({"class":"downsize", "sentence":"what do you think about downsizing?"})
training_data.append({"class":"downsize", "sentence":"tell me about downsizing"})

training_data.append({"class":"bored", "sentence":"I'm bored"})
training_data.append({"class":"bored", "sentence":"I am bored"})
training_data.append({"class":"bored", "sentence":"entertain me"})
training_data.append({"class":"bored", "sentence":"so boring"})
training_data.append({"class":"bored", "sentence":"so bored"})
training_data.append({"class":"bored", "sentence":"I'm sad"})
training_data.append({"class":"bored", "sentence":"I am sad"})

training_data.append({"class":"compliment", "sentence":"you’re funny"})
training_data.append({"class":"compliment", "sentence":"you are funny"})
training_data.append({"class":"compliment", "sentence":"you’re hilarious"})
training_data.append({"class":"compliment", "sentence":"you are hilarious"})
training_data.append({"class":"compliment", "sentence":"I like you"})
training_data.append({"class":"compliment", "sentence":"that's a good idea"})
training_data.append({"class":"compliment", "sentence":"that is a good idea"})
training_data.append({"class":"compliment", "sentence":"that's a great idea"})
training_data.append({"class":"compliment", "sentence":"that is a great idea"})
training_data.append({"class":"compliment", "sentence":"I like your idea"})
training_data.append({"class":"compliment", "sentence":"I love your idea"})
training_data.append({"class":"compliment", "sentence":"I like the idea"})
training_data.append({"class":"compliment", "sentence":"I love the idea"})
training_data.append({"class":"compliment", "sentence":"you look cute today, Dwight"})


training_data.append({"class":"laughing", "sentence":"haha"})
training_data.append({"class":"laughing", "sentence":"hahaha"})
training_data.append({"class":"laughing", "sentence":"lol"})
training_data.append({"class":"laughing", "sentence":"that's funny"})
training_data.append({"class":"laughing", "sentence":"that is funny"})
training_data.append({"class":"laughing", "sentence":"that is so funny"})

training_data.append({"class":"instructions", "sentence":"what should I ask you about?"})
training_data.append({"class":"instructions", "sentence":"what should we talk about?"})
training_data.append({"class":"instructions", "sentence":"what do you want to talk about?"})
training_data.append({"class":"instructions", "sentence":"I don't know what to talk about"})
training_data.append({"class":"instructions", "sentence":"I don't know what to ask you"})
training_data.append({"class":"instructions", "sentence":"do you know what to talk about?"})

training_data.append({"class":"dunder_mifflin", "sentence":"what do you think about dunder mifflin?"})
training_data.append({"class":"dunder_mifflin", "sentence":"do you like working at dunder mifflin?"})

training_data.append({"class":"jim", "sentence":"what do you think about jim?"})
training_data.append({"class":"jim", "sentence":"jim"})
training_data.append({"class":"jim", "sentence":"do you like jim?"})

training_data.append({"class":"michael", "sentence":"what do you think about michael?"})
training_data.append({"class":"michael", "sentence":"michael"})
training_data.append({"class":"michael", "sentence":"do you like michael?"})
training_data.append({"class":"michael", "sentence":"is micheal a good boss?"})
training_data.append({"class":"michael", "sentence":"is micheal a bad boss?"})
training_data.append({"class":"michael", "sentence":"how is micheal as a boss?"})
training_data.append({"class":"michael", "sentence":"is micheal a good friend?"})
training_data.append({"class":"michael", "sentence":"is michael a bad friend?"})
training_data.append({"class":"michael", "sentence":"how is micheal as a friend?"})

training_data.append({"class":"angela", "sentence":"what do you think about Angela?"})

training_data.append({"class":"andy", "sentence":"what do you think about Andy?"})

training_data.append({"class":"insult", "sentence":"you're mean"})
training_data.append({"class":"insult", "sentence":"you're rude"})
training_data.append({"class":"insult", "sentence":"you're ugly"})
training_data.append({"class":"insult", "sentence":"you're so mean"})
training_data.append({"class":"insult", "sentence":"you're so rude"})
training_data.append({"class":"insult", "sentence":"you're so ugly"})
training_data.append({"class":"insult", "sentence":"mean"})
training_data.append({"class":"insult", "sentence":"rude"})
training_data.append({"class":"insult", "sentence":"ugly"})
training_data.append({"class":"insult", "sentence":"stupid"})

training_data.append({"class":"overpopulation", "sentence":"overpopulation"})
training_data.append({"class":"overpopulation", "sentence":"we should we do about overpopulation?"})
training_data.append({"class":"overpopulation", "sentence":"overpopulation is an issue"})

training_data.append({"class":"where", "sentence":"where is your stapler?"})
training_data.append({"class":"where", "sentence":"where is your bobblehead?"})
training_data.append({"class":"where", "sentence":"where is your lunch?"})
training_data.append({"class":"where", "sentence":"where is my stapler?"})
training_data.append({"class":"where", "sentence":"where is my bobblehead?"})
training_data.append({"class":"where", "sentence":"where is my lunch?"})

training_data.append({"class":"checking_up", "sentence":"you okay?"})
training_data.append({"class":"checking_up", "sentence":"you good?"})
training_data.append({"class":"checking_up", "sentence":"you doing okay?"})
training_data.append({"class":"checking_up", "sentence":"are you okay?"})
training_data.append({"class":"checking_up", "sentence":"are you good?"})
training_data.append({"class":"checking_up", "sentence":"are you doing okay?"})

training_data.append({"class":"appearances", "sentence":"how do I look?"})
training_data.append({"class":"appearances", "sentence":"do I look okay?"})
training_data.append({"class":"appearances", "sentence":"do I look good?"})

training_data.append({"class":"day", "sentence":"what day is it?"})
training_data.append({"class":"day", "sentence":"what is today?"})
training_data.append({"class":"day", "sentence":"is today Thursday?"})
training_data.append({"class":"day", "sentence":"is today Friday?"})

training_data.append({"class":"birthday", "sentence":"it's my birthday!"})
training_data.append({"class":"birthday", "sentence":"today is my birthday!"})

training_data.append({"class":"cholesterol", "sentence":"why would you want to raise your cholesterol?"})

training_data.append({"class":"noanswer", "sentence":"  "})
print ("%s sentences of training data" % len(training_data))

# capture unique stemmed words in the training corpus
corpus_words = {}
class_words = {}
# turn a list into a set (of unique items) and then a list again (this removes duplicates)
classes = list(set([a['class'] for a in training_data]))
for c in classes:
    # prepare a list of words within each class
    class_words[c] = []

# loop through each sentence in our training data
for data in training_data:
    # tokenize each sentence into words
    for word in nltk.word_tokenize(data['sentence']):
        # ignore a some things
        if word not in ["?", "'s"]:
            # stem and lowercase each word
            stemmed_word = stemmer.stem(word.lower())
            # have we not seen this word already?
            if stemmed_word not in corpus_words:
                corpus_words[stemmed_word] = 1
            else:
                corpus_words[stemmed_word] += 1

            # add the word to our words in class list
            class_words[data['class']].extend([stemmed_word])

# we now have each stemmed word and the number of occurances of the word in our training corpus (the word's commonality)
print ("Corpus words and counts: %s \n" % corpus_words)
# also we have all words in each class
print ("Class words: %s" % class_words)

# calculate a score for a given class taking into account word commonality
def calculate_class_score(sentence, class_name, show_details=True):
    score = 0
    # tokenize each word in our new sentence
    for word in nltk.word_tokenize(sentence):
        # check to see if the stem of the word is in any of our classes
        if stemmer.stem(word.lower()) in class_words[class_name]:
            # treat each word with relative weight
            score += (1 / corpus_words[stemmer.stem(word.lower())])

            if show_details:
                print ("   match: %s (%s)" % (stemmer.stem(word.lower()), 1 / corpus_words[stemmer.stem(word.lower())]))
    return score

# return the class with highest score for sentence
def classify(sentence):
    high_class = None
    high_score = 0
    # loop through our classes
    for c in class_words.keys():
        # calculate score of sentence for each class
        score = calculate_class_score(sentence, c, show_details=False)
        # keep track of highest score
        if score > high_score:
            high_class = c
            high_score = score

    return high_class
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence


def getResponse(ints, intents_json):
	tag = ints
	if(ints == None):
		tag="noanswer"
	print(type(tag))
	list_of_intents = intents_json['intents']
	for i in list_of_intents:
		if(i['tag']== tag):
			result = random.choice(i['responses'])
			break
	return result
    #if(ints == None):
		#tag="noanswer"
def chatbot_response(msg):
    ints = classify(msg)
    res = getResponse(ints, intents)
    return res


#Creating GUI with tkinter
import tkinter
from tkinter import *


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title("Hello")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()
