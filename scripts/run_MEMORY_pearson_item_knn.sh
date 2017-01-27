#!/bin/bash
#
#python holdout_eval.py \
#    ../data/ml100k/ratings.csv \
#    --header 0 \
#    --recommender MEMORY_item_knn \
#    --params similarity=pearson,k=50,shrinkage=100 \
#    --rec_length 100 \
#    --prediction_file ../results/MOVIES_MEMORY_pearson_item_knn.txt

python holdout_eval.py \
    ../data/competition/interactions.csv \
    --header 0 \
    --recommender MEMORY_item_knn \
    --params similarity=pearson,k=50,shrinkage=100 \
    --rec_length 5 \
    --sep ! \
    --user_key user_id \
    --item_key item_id \
    --rating_key interaction_type \
    --prediction_file ../results/Pearson_K50_Shrinkage100.csv \
    --target_user ../data/competition/tu.csv