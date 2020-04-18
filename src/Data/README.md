This directory contains all the code for extracting data from various sources.

The individual ipython notebooks are linked to Google Colab and must be run on TPU's.

The dataset for the movie plots and posters was extracted using AWS Lambda functions provided in the AWS_Lambda directory.
There are two lambdas, one each for poster image and plot summary/genres, that need IMDB movieid as inputs. The movieid typivcally has the form 'ttXXXXX'.

The movieid file is dropped into an S3 bucket and subsequently a SQS queue is genrated for each of this movieids. The SQS queue feeds the movieid to both the lambdas (getIMDBMetadata and getIMDBPosters) and the results are stored in an output S3 bucket.
