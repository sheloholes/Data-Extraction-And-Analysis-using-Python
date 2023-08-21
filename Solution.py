#!/usr/bin/env python
# coding: utf-8

# # Data Extraction

# In[9]:


import pandas as pd
import bs4
import requests


# In[52]:


df=pd.read_csv("Input.xlsx.csv")          # To read input file
df['URL_ID'] = df['URL_ID'].astype(str).replace('\.0', '', regex=True) #For accurate file name
print("Dataframe Created")


# In[55]:


#This Block of Code Takes Time To Complete
import time

start = time.time()

print("Extraction Started")
for index, row in df.iterrows():
    result=requests.get(row["URL"])
    soup=bs4.BeautifulSoup(result.text,'lxml')
    try:                                                          #Using Try try block for error handeling
        title=soup.select("h1")[0].text
        filecontent=soup.select(".td-post-content")[0].text
    except:
        filecontent="Ooops... Error 404"
    with open(f".//output_files//{row['URL_ID']}.txt", "w", encoding="utf-8") as f:
        f.write(title)
        f.write(filecontent)

end = time.time()



print("Extraction Complete")
print("Time taken for extraction ",end-start)


# # For Data Analysis

# In[56]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import punkt
from textblob import TextBlob
from nltk.corpus import cmudict
# Download if not already downloaded
# nltk.download('vader_lexicon')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('corpus')


#Output Dataframe
output_df = pd.DataFrame(
    columns=['URL_ID','URL','POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE',
             'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE',
             'COMPLEX WORD COUNT', 'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH',])

def count_syllables(word, dic):
    return len(dic.inserted(word).split('-'))


#Iterating input.csv file's df for reference
for index, row in df.iterrows():
    output_df.loc[index,"URL_ID"]=row["URL_ID"]
    output_df.loc[index,"URL"]=row["URL"]
    with open(f".//output_files//{row['URL_ID']}.txt", "r", encoding="utf-8") as f:
        data=f.read()
        
        sia = SentimentIntensityAnalyzer()
        # Get sentiment scores
        sentiment_scores = sia.polarity_scores(data)
        # Positive score
        output_df.loc[index,"POSITIVE SCORE"] = sentiment_scores['pos']
        output_df.loc[index,'NEGATIVE SCORE'] = sentiment_scores['neg']
        
          
        
        blob = TextBlob(data)
        output_df.loc[index,'POLARITY SCORE'] = blob.sentiment.polarity
        output_df.loc[index,'SUBJECTIVITY SCORE'] = blob.sentiment.subjectivity
        
        
        sentences = nltk.sent_tokenize(data)
        words = nltk.word_tokenize(data)
        
        
        
        
        # Count total number of words
        total_words = sum(len(nltk.word_tokenize(sentence)) for sentence in sentences)
        # Count total number of sentences
        total_sentences = len(sentences)
        # Calculate average sentence length
        avg_sent_length=total_words / total_sentences
        
        output_df.loc[index,'AVG SENTENCE LENGTH'] = avg_sent_length
        
        output_df.loc[index,'AVG NUMBER OF WORDS PER SENTENCE'] = avg_sent_length
    
        # Initialize pyphen dictionary
        dic = pyphen.Pyphen(lang='en')

        # Define a threshold for complex words (e.g., words with more than three syllables)
        complex_word_threshold = 3

        # Process text in smaller chunks
        chunk_size = 100  # Adjust as needed
        complex_word_count = sum(1 for word in words if count_syllables(word, dic) > complex_word_threshold)
        total_word_count = len(words)

        for i in range(0, len(words), chunk_size):
            chunk = words[i:i+chunk_size]
            for word in chunk:
                total_word_count += 1
                if count_syllables(word, dic) > complex_word_threshold:
                    complex_word_count += 1

        # Calculate the percentage of complex words
        percent_of_complex_words=(complex_word_count / total_word_count) * 100
        
        output_df.loc[index,'PERCENTAGE OF COMPLEX WORDS'] = percent_of_complex_words
        output_df.loc[index,'COMPLEX WORD COUNT'] = complex_word_count
        output_df.loc[index,'WORD COUNT'] = total_word_count

        FOG_Index = 0.4 * (avg_sent_length + percent_of_complex_words)
        
        output_df.loc[index,'FOG INDEX']=FOG_Index
        
        
        
        total_syllables = sum(count_syllables(word, dic) for word in words)
        average_syllables_per_word = total_syllables / total_words
        output_df.loc[index,'SYLLABLE PER WORD']=average_syllables_per_word
        
        
        personal_pronouns = ["I", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"]

        # Count the occurrences of personal pronouns
        personal_pronoun_count = sum(1 for word in words if word.lower() in personal_pronouns)
        output_df.loc[index,'PERSONAL PRONOUNS']=personal_pronoun_count
        
        
        
        total_characters = sum(len(word) for word in words)
        average_word_length = total_characters / total_words
        output_df.loc[index,'AVG WORD LENGTH']=average_word_length
print("Analysis Complete")


# In[57]:


print(output_df.head())

