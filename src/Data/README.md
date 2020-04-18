# Data

This directory contains all the code for extracting data from various sources.

**NOTE**: The individual ipython notebooks are linked to Google Colab and must be run on TPU's.

The dataset for the movie plots and posters was extracted using AWS Lambda functions provided in the AWS_Lambda directory.
There are two lambdas, one for the poster image and the other for plot summary/genres, that need the IMDB movieid as inputs. The movieid typically has the format `ttXXXXX`. The movieid file is then dropped into an S3 bucket and subsequently a SQS queue is generated for each of the movieid's. The SQS queue feeds the movieid to both the lambdas (`getIMDBMetadata` and `getIMDBPosters`) and the results are stored in an output S3 bucket.
