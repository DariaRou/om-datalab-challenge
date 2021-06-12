
import pandas as pd
import numpy as np
import os

def get_data():
    # define file path, data stored in "/om-datalab-challenge/om-datalab-challenge/data"
    filepath = 'data/french_tweets.csv'
    # get the a sample data in a dataFrame
    french = pd.read_csv(filepath).sample(5000).reset_index(drop=True)
