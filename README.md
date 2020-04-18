# An Integrated Visio-Textual Approach to Movie Genre Classification

**Authors** : Sartaj Singh Baveja, Karthik Srinivasan

## Abstract

As movie collections within streaming services grow, providing relevant movie recommendations have become an integral part for customer satisfaction and retention. For this purpose, classification of movies into genres have taken center stage. In this project, we describe a novel method of combining the movie plot summaries with their poster publications to better describe its genre. We propose a multi-headed deep neural network (DNN) that simultaneously trains the plot summaries and the poster images, while optimizing for maximum F1 scores. We show that the results from this combined DNN out-perform conventional Natural Language Processing (NLP) and convolutional networks (CNN).

## Code

**Note:** All the codes in this repository were run on TPU's with High RAM on Google Colab and tensorflow version 2.1.0. 

Our code is structured in the following way. 

- [src/Data](./src/Data) folder contains the data related ipython notebooks where we source and parse the data, clean it and generate BERT tokens as well as poster feature vectors along with perform EDA on it
- [src/Baseline BERT](./src/Baseline%20BERT) folder contains the ipython notebook for our baseline BERT model
- [src/Baseline Posters](./src/Baseline%20Posters) folder contains the ipython notebook for our baseline Poster model
- [src/Combined Model](./src/Combined%20Model) folder contains the ipython notebook for our multi-headed deep neural network integrated model
- [src/Analysis](./src/Analysis) folder contains ipython notebooks for our error analysis as well as weighted f1 score analysis for different models
- [Final Submission](./Final%20Submission) folder contains our Final Presentation along with the Final Paper that was submitted.
