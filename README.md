# w266 Final Project

**Project** : An Integrated Visio-Textual Approach to Movie Genre Classification <br />
**Authors** : Sartaj Singh Baveja, Karthik Srinivasan

## Abstract

As movie collections within streaming services grow, providing relevant movie recommendations have become an integral part for customer satisfaction and retention. For this purpose, classification of movies into genres have taken center stage. In this project, we describe a novel method of combining the movie plot summaries with their poster publications to better describe its genre. We propose a multi-headed deep neural network (DNN) that simultaneously trains the plot summaries and the poster images, while optimizing for maximum F1 scores. We show that the results from this combined DNN out-perform conventional Natural Language Processing (NLP) and convolutional networks (CNN).

## Project Structure

Our project is structured in the following way. 

- `src/Data` folder contains the data related notebooks where we source and parse the data, clean it and generate BERT tokens as well as poster feature vectors along with perform EDA on it
- `src/Baseline BERT` folder contains our baseline BERT model
- `src/Baseline Posters` folder contains our baseline Poster model
- `src/Combined Model` folder contains our multi-headed deep neural network integrated model
- `src/Analysis` folder contains notebooks where we perform error analysis as well as calculate our weighted f1 scores for different models
- `Final Submission` folder contains our Final Presentation along with the Final Paper that was submitted.
