#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import pandas as pd
import numpy as np
from operator import itemgetter 
from subword_nmt import apply_bpe
from subword_nmt.learn_bpe import learn_bpe





#General Preprocessing using stopwords, lowercasing, and removing punctuation

with open("stopwords.txt", "r") as f:
    stopwords = f.read().splitlines() 

#Read in the training tsv
train = pd.read_table("train.tsv")

#Lower case and remove punctuation
train["Phrase"] = train["Phrase"].str.lower().str.replace('[^\w\s]','') 
#Take out stopwords
train['Phrase'] = train['Phrase'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)])) 




#Choose whether to enable BPE Subwording preprocessing

#enableBPE = True
enableBPE = False


if(enableBPE):
    #Subword preprocessing
    
    
    #Get the subwords based on the training data
    outfile=open("learnBPE.txt","w")
    learn_bpe(train["Phrase"],outfile, 100000, 4)

    
    #Apply BPE to the learned subwords
    infile=open("learnBPE.txt","r")
    bpe = apply_bpe.BPE(infile)
    
    #Add the subwords to the vocabulary
    for index, phrase in train['Phrase'].items():
        train.loc[index, 'Phrase'] +=  bpe.process_line(phrase)
 
        
    #Remove the added '@@' from each subword
    train["Phrase"] = train["Phrase"].str.replace('[^\w\s]','') 




#Map from 5 value to 3 value sentiments and store in a new column
train["ThreeScale"] = pd.cut(np.array(train["Sentiment"]), 3, labels=["negative", "neutral", "positive"]) 




#------------THREE VALUE SCALE---------------



#Assign the reviews by sentiment values to variables for inspection (3-Value Scale)
threeScalePos = train[train.ThreeScale == "positive"]
threeScaleNeu = train[train.ThreeScale == "neutral"]
threeScaleNeg = train[train.ThreeScale == "negative"]


#Count the occurences of each feature in each class (3-Value Scale)
positiveFeatureDict = dict()
for index, phrase in threeScalePos.Phrase.items():
    words = phrase.split()
    
    for word in words:
        try:
            positiveFeatureDict[word] += 1
        except:
            positiveFeatureDict[word] = 1

    
neutralFeatureDict = dict()
for index, phrase in threeScaleNeu.Phrase.items():
    words = phrase.split()
    
    for word in words:
        try:
            neutralFeatureDict[word] += 1
        except:
            neutralFeatureDict[word] = 1

    

negativeFeatureDict = dict()
for index, phrase in threeScaleNeg.Phrase.items():
    words = phrase.split()
    
    for word in words:
        try:
            negativeFeatureDict[word] += 1
        except:
            negativeFeatureDict[word] = 1





#Choose whether to enable the custom model
#customModel = True
customModel = False




    
if(customModel):
    
    #Get the top 7 words from each class
    topPos = dict(sorted(positiveFeatureDict.items(), key = itemgetter(1), reverse = True)[:7]) 
    topNeu= dict(sorted(neutralFeatureDict.items(), key = itemgetter(1), reverse = True)[:7]) 
    topNeg = dict(sorted(negativeFeatureDict.items(), key = itemgetter(1), reverse = True)[:7]) 

    


    #If they appear in any other top 7, remove them as they aren't descriptive enough
    for word in topPos:
        if word in topNeu:
            del positiveFeatureDict[word]
        elif word in topNeg:
            del positiveFeatureDict[word]


    for word in topNeu:
        if word in topPos:
            del neutralFeatureDict[word]
        elif word in topNeg:
            del neutralFeatureDict[word]

 
    for word in topNeg:
        if word in topNeu:
            del negativeFeatureDict[word]
        elif word in topPos:
            del negativeFeatureDict[word]  
            
            
            
 


#Count the total training vocabulary size for use in the classifier later
wordDict = dict()
for index, phrase in train['Phrase'].items():
    words = phrase.split()
    
    for word in words:
        try:
            wordDict[word] += 1
        except:
            wordDict[word] = 1
            
            
            
            
       
#Printing information about the training data  (3-VALUE SCALE)          
print("\n", "------------------------------------------------", "\n")
print("Vocabulary Size :", len(wordDict), "\n")
              
                
                

print("\n", "------------------------------------------------", "\n")
print("THREE VALUE SCALE BREAKDOWN", "\n", "\n")


print("Positives Phrases:" , len(threeScalePos), "\n")
print("Neutrals Phrases:" ,len(threeScaleNeu), "\n")
print("Negatives Phrases:" ,len(threeScaleNeg), "\n")


#Update the top 10 list
tenPos = dict(sorted(positiveFeatureDict.items(), key = itemgetter(1), reverse = True)[:10]) 
tenNeu= dict(sorted(neutralFeatureDict.items(), key = itemgetter(1), reverse = True)[:10]) 
tenNeg = dict(sorted(negativeFeatureDict.items(), key = itemgetter(1), reverse = True)[:10]) 

print("Relevant Positive Features :", tenPos, "\n")
print("Relevant Neutral Features :", tenNeu, "\n")
print("Relevant Negative Features :", tenNeg, "\n")  
    




#------------FIVE VALUE SCALE---------------



#Assign the reviews by sentiment values to variables for inspection (5-Value Scale)
fiveScaleZero = train[train.Sentiment == 0]
fiveScaleOne = train[train.Sentiment == 1]
fiveScaleTwo = train[train.Sentiment == 2]
fiveScaleThree = train[train.Sentiment == 3]
fiveScaleFour = train[train.Sentiment == 4]


#Count the occurences of each feature in each class (5-Value Scale)
zeroFeatureDict = dict()
for index, phrase in fiveScaleZero.Phrase.items():
    words = phrase.split()
    
    for word in words:
        try:
            zeroFeatureDict[word] += 1
        except:
            zeroFeatureDict[word] = 1




oneFeatureDict = dict()
for index, phrase in fiveScaleOne.Phrase.items():
    words = phrase.split()
    
    for word in words:
        try:
            oneFeatureDict[word] += 1
        except:
            oneFeatureDict[word] = 1



twoFeatureDict = dict()
for index, phrase in fiveScaleTwo.Phrase.items():
    words = phrase.split()
    
    for word in words:
        try:
            twoFeatureDict[word] += 1
        except:
            twoFeatureDict[word] = 1


threeFeatureDict = dict()
for index, phrase in fiveScaleThree.Phrase.items():
    words = phrase.split()
    
    for word in words:
        try:
            threeFeatureDict[word] += 1
        except:
            threeFeatureDict[word] = 1



fourFeatureDict = dict()
for index, phrase in fiveScaleFour.Phrase.items():
    words = phrase.split()
    
    for word in words:
        try:
            fourFeatureDict[word] += 1
        except:
            fourFeatureDict[word] = 1
            
            
            
  

if(customModel):
    
    topZero = dict(sorted(zeroFeatureDict.items(), key = itemgetter(1), reverse = True)[:7]) 
    topOne= dict(sorted(oneFeatureDict.items(), key = itemgetter(1), reverse = True)[:7]) 
    topTwo = dict(sorted(twoFeatureDict.items(), key = itemgetter(1), reverse = True)[:7]) 
    topThree= dict(sorted(threeFeatureDict.items(), key = itemgetter(1), reverse = True)[:7]) 
    topFour = dict(sorted(fourFeatureDict.items(), key = itemgetter(1), reverse = True)[:7]) 
        
    

    for word in topZero:
        if word in topOne:
            del zeroFeatureDict[word]
        elif word in topTwo:
            del zeroFeatureDict[word]
        elif word in topThree:
            del zeroFeatureDict[word]
        elif word in topFour:
            del zeroFeatureDict[word]


    for word in topOne:
        if word in topZero:
            del oneFeatureDict[word]
        elif word in topTwo:
            del oneFeatureDict[word]
        elif word in topThree:
            del oneFeatureDict[word]
        elif word in topFour:
            del oneFeatureDict[word]


    for word in topTwo:
        if word in topOne:
            del twoFeatureDict[word]
        elif word in topZero:
            del twoFeatureDict[word]
        elif word in topThree:
            del twoFeatureDict[word]
        elif word in topFour:
            del twoFeatureDict[word]
            
            
    for word in topThree:
        if word in topOne:
            del threeFeatureDict[word]
        elif word in topTwo:
            del threeFeatureDict[word]
        elif word in topZero:
            del threeFeatureDict[word]
        elif word in topFour:
            del threeFeatureDict[word]
            
            
    for word in topFour:
        if word in topOne:
            del fourFeatureDict[word]
        elif word in topTwo:
            del fourFeatureDict[word]
        elif word in topThree:
            del fourFeatureDict[word]
        elif word in topZero:
            del fourFeatureDict[word]
            

#Printing information about the training data  (5-VALUE SCALE)  

print("\n", "------------------------------------------------", "\n")
print("FIVE VALUE SCALE BREAKDOWN", "\n", "\n")

print("Zero Score Phrases:" , len(fiveScaleZero), "\n")
print("One Score Phrases:" ,len(fiveScaleOne), "\n")
print("Two Score Phrases:" ,len(fiveScaleTwo), "\n")
print("Three Score Phrases:" ,len(fiveScaleThree), "\n")
print("Four Score Phrases:" ,len(fiveScaleFour), "\n")


tenZero = dict(sorted(zeroFeatureDict.items(), key = itemgetter(1), reverse = True)[:10]) 
tenOne = dict(sorted(oneFeatureDict.items(), key = itemgetter(1), reverse = True)[:10]) 
tenTwo = dict(sorted(twoFeatureDict.items(), key = itemgetter(1), reverse = True)[:10]) 
tenThree = dict(sorted(threeFeatureDict.items(), key = itemgetter(1), reverse = True)[:10]) 
tenFour = dict(sorted(fourFeatureDict.items(), key = itemgetter(1), reverse = True)[:10]) 



print("Relevant Zero Score Features :", tenZero, "\n")
print("Relevant One Score Features :", tenOne, "\n") 
print("Relevant Two Score Features :", tenTwo, "\n")  
print("Relevant Three Score Features :", tenThree, "\n")
print("Relevant Four Score Features :", tenFour, "\n")  




#------------THREE VALUE BAYES CLASSIFIER--------------


def NaiveBayesThree (phrase):
    
    allPhrases = train.Sentiment.size
    
    positivePhrases = threeScalePos.size
    neutralPhrases = threeScaleNeu.size
    negativePhrases = threeScaleNeg.size
    
    #compute prior probability of each class
    positiveProbability = positivePhrases/ allPhrases
    neutralProbability = neutralPhrases/ allPhrases
    negativeProbability = negativePhrases/ allPhrases
    


    
    #Compute the likelihood for all features in that phrase
    for feature in phrase.split():
    
        try:
            posVal = (positiveFeatureDict[feature] + 1) / (len(positiveFeatureDict) + len(wordDict))
        except:
            posVal = 1 / (len(positiveFeatureDict) + len(wordDict))
        
        
        try:
            neuVal = (neutralFeatureDict[feature] + 1) / (len(neutralFeatureDict) + len(wordDict))
        except:
            neuVal = 1 / (len(neutralFeatureDict) + len(wordDict))
            
        try:   
            negVal = (negativeFeatureDict[feature] + 1) / (len(negativeFeatureDict) + len(wordDict))
        except: 
            negVal = 1 / (len(negativeFeatureDict) + len(wordDict))
    
        

        #Multiply the corresponding probabilities and store the new values
        positiveProbability *= posVal
    
        neutralProbability *= neuVal
    
        negativeProbability *= negVal
        
    
    #Declare the winner by highest probability
    winner = max(positiveProbability, neutralProbability, negativeProbability)
    
    
    if winner == positiveProbability:
        return "positive"
    
    elif winner == neutralProbability:
        return "neutral"
    
    elif winner == negativeProbability:
        return "negative"
    




#------------FIVE VALUE BAYES CLASSIFIER--------------




def NaiveBayesFive (phrase):
    
    allPhrases = train.Sentiment.size
    
    zeroPhrases = fiveScaleZero.size
    onePhrases = fiveScaleOne.size
    twoPhrases = fiveScaleTwo.size
    threePhrases = fiveScaleThree.size
    fourPhrases = fiveScaleFour.size
    
    
    
    #compute prior probability of each class
    zeroProbability = zeroPhrases/ allPhrases
    
    oneProbability = onePhrases/ allPhrases
    
    twoProbability = twoPhrases/ allPhrases
    
    threeProbability = threePhrases/ allPhrases
    
    fourProbability = fourPhrases/ allPhrases
    
    
    
    
    #Compute the likelihood for all features in that phrase
    for feature in phrase.split():
    
        try:
            zeroVal = (zeroFeatureDict[feature] + 1) / (len(zeroFeatureDict) + len(wordDict))
        except:
            zeroVal = 1 / (len(zeroFeatureDict) + len(wordDict))
        
        
        try:
            oneVal = (oneFeatureDict[feature] + 1) / (len(oneFeatureDict) + len(wordDict))
        except:
            oneVal = 1 / (len(oneFeatureDict) + len(wordDict))
        
            
        try:
            twoVal = (twoFeatureDict[feature] + 1) / (len(twoFeatureDict) + len(wordDict))
        except:
            twoVal = 1 / (len(twoFeatureDict) + len(wordDict))
        
        try:
            threeVal = (threeFeatureDict[feature] + 1) / (len(threeFeatureDict) + len(wordDict))
        except:
            threeVal = 1 / (len(threeFeatureDict) + len(wordDict))
        
        
        try:
            fourVal = (fourFeatureDict[feature] + 1) / (len(fourFeatureDict) + len(wordDict))
        except:
            fourVal = 1 / (len(fourFeatureDict) + len(wordDict))
        
    
        #Multiply the corresponding probabilities and store the new values
        zeroProbability *= zeroVal

        oneProbability *= oneVal
        
        twoProbability *= twoVal
        
        threeProbability *= threeVal
                
        fourProbability *= fourVal
        
    
    #Declare the winner by highest probability
    winner = max(zeroProbability, oneProbability, twoProbability,threeProbability, fourProbability )
    
    
    if winner == zeroProbability:
        return 0
    
    elif winner == oneProbability:
        return 1
    
    elif winner == twoProbability:
        return 2
    
    elif winner == threeProbability:
        return 3
    
    else :
        return 4
    

    return








dev = pd.read_table("dev.tsv")
dev["Phrase"] = dev["Phrase"].str.lower().str.replace('[^\w\s]','') #Lower case and remove punctuation
dev['Phrase'] = dev['Phrase'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)])) #Take out stopwords
write = 'dev_predictions_3classes_Cian_MORIARTY.tsv'


    
with open(write, 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['SentenceId', 'Sentiment'])
    
    
    for index, phrase in dev['Phrase'].items():
        
        sentenceId = index
        sentiment = NaiveBayesThree (phrase)
        tsv_writer.writerow([sentenceId, sentiment])   
        
        
        
        

write = 'dev_predictions_5classes_Cian_MORIARTY.tsv'

with open(write, 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['SentenceId', 'Sentiment'])
    
    
    for index, phrase in dev['Phrase'].items():
        
        sentenceId = index
        sentiment = NaiveBayesFive (phrase)
        tsv_writer.writerow([sentenceId, sentiment])     
    



def CalculateCorrectness(guesses, actual):
    score = 0
    
    for i in range(len(actual)):
        if guesses[i] == actual[i]:
            score += 1
            
    

    return score/len(actual)*100




fiveClassDevPredictions = pd.read_table("dev_predictions_5classes_Cian_MORIARTY.tsv")
threeClassDevPredictions = pd.read_table("dev_predictions_3classes_Cian_MORIARTY.tsv")


#Calculate score for dev 3 class
dev["ThreeScale"] = pd.cut(np.array(dev["Sentiment"]), 3, labels=["negative", "neutral", "positive"]) 
print("Three-Scale Classifier Correctness:" , CalculateCorrectness(threeClassDevPredictions["Sentiment"], dev["ThreeScale"]))

#Calculate score for dev 5 class
print("Five-Scale Classifier Correctness:" , CalculateCorrectness(fiveClassDevPredictions["Sentiment"], dev["Sentiment"]))














read = pd.read_table("test.tsv")
write = 'test_predictions_3classes_Cian_MORIARTY.tsv'


    
with open(write, 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['SentenceId', 'Sentiment'])
    
    
    for index, phrase in read['Phrase'].items():
        
        sentenceId = index
        sentiment = NaiveBayesThree (phrase)
        tsv_writer.writerow([sentenceId, sentiment])      
        
        
        
        
        
write = 'test_predictions_5classes_Cian_MORIARTY.tsv'

    
with open(write, 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['SentenceId', 'Sentiment'])
    
    
    for index, phrase in read['Phrase'].items():
        
        sentenceId = index
        sentiment = NaiveBayesFive (phrase)
        tsv_writer.writerow([sentenceId, sentiment])     
    
    
    
    
    


    
