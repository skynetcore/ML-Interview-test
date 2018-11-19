# Bonsai Interview Test

Hi this is Derric taking Bonsai ML test

### Understood the Goal:
I have understood that the task is to recommend a book that a user is most likely to buy next using the data provided. I have taken the dataset from here: https://www.dropbox.com/sh/uj3nsf66mtwm36q/AADLUNVShEZ0VI3DsLad6S4Ta?dl=0
The model is being made feasible in a production environment considering algorithm, speed and accuracy.

The time for generating suggestion books from 3000 available books per user is currently 40 minutes

### Approach:
The final includes all relevant codes used, and an output csv file. The csv file  a 
M x N matrix, where M is the number of users and N is the number of products, where each entry r<sub>ij</sub> 
represents the rating of product j for a given user i.

I have used a rating model that that is from 0 - 1.0, with 0.5 being the probable chance that a user will buy the book.

