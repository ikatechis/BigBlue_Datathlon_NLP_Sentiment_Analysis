### BigBlue_Datathlon_NLP_Sentiment_Analysis
# Game Reviews Sentiment NLP Analysis
The Datathlon from Big Blue Data Academy where we had 8 hours to perform sentiment analysis on game reviews using ML and DL techniques.

We tried various preprocessing techniques: stopwords removal, pos tagging, regex cleaning and applied different LSTM Neural network architectures. Finally, the highest accuracy we got from tranfer learning using tensorflow's [nnlm-en-dim50](https://tfhub.dev/google/nnlm-en-dim50/1) model which we retrained with our cleaned dataset. 

We found that stop-word removal and POS cleaning did not have any effect on accuracy which by the end of datathlon was at 88%.

Furthermore, we performed spell and profanity checking, emojies and emoticon analysis and finally delivered the classifier in a command-line-based UI that performed tha classification on unseen test data and delivered the results.

### Results

For our full final presentation [click here](https://docs.google.com/presentation/d/1Ko0SFnLkGr_n_O9Ht8MAeGP69II89Btwbh88v0RL9NY/edit?usp=sharing)


![slide1](https://user-images.githubusercontent.com/17815370/155307744-8f2337f4-ae79-4664-8761-a3a156ff7a1f.png)
![slide2](https://user-images.githubusercontent.com/17815370/155307754-288e5010-ce40-4595-9e9e-4f5871224021.png)
![slide3](https://user-images.githubusercontent.com/17815370/155307761-d45e817c-6d22-492e-b150-2a46c0b25220.png)
