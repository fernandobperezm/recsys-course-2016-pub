#!/bin/bash

#python holdout_eval.py ../data/ml100k/ratings.csv --header 0 --make_binary --binary_th 4 \
# --recommender item_knn --params similarity=cosine,k=50,shrinkage=25,normalize=False

#python new_user_eval.py ../data/ml100k/ratings.csv --header 0 --make_binary --binary_th 4 \
#--n_observed 3 --recommender item_knn --params similarity=cosine,k=50,shrinkage=25,normalize=False

# Binary ratings for MOVIES.
#python holdout_eval.py \
#    ../data/ml100k/ratings.csv \
#    --header 0 \
#    --make_binary \
#    --binary_th 4 \
#    --recommender MEMORY_item_knn \
#    --params similarity=adj-cosine,k=50,shrinkage=25 \
#    --rec_length 5 \
#    --prediction_file ../results/MOVIES_BINARY_MEMORY_adjustedcosine_K50_Shrinkage25.csv

### Not binary ratings.
#python holdout_eval.py \
#    ../data/competition/interactions.csv \
#    --header 0 \
#    --recommender MEMORY_item_knn \
#    --params similarity=adj-cosine,k=50,shrinkage=100 \
#    --rec_length 5 \
#    --sep ! \
#    --user_key user_id \
#    --item_key item_id \
#    --rating_key interaction_type \
#    --prediction_file ../results/adjustedcosine_K50_Shrinkage100.csv \
#    --target_user ../data/competition/tu.csv
#
## Binary ratings.
python holdout_eval.py \
    ../data/competition/interactions.csv \
    --header 0 \
    --make_binary \
    --binary_th 1 \
    --recommender MEMORY_item_knn \
    --params similarity=adj-cosine,k=5,shrinkage=100 \
    --rec_length 5 \
    --sep ! \
    --user_key user_id \
    --item_key item_id \
    --rating_key interaction_type \
    --prediction_file ../results/adjustedcosine_binary_K5_Shrinkage100.csv \
    --target_user ../data/competition/tu.csv