import pickle
import pandas as pd

#load model
with open('30_diamond_model_pkl', 'rb') as f:
    model = pickle.load(f)

#let's see if we can make a prediction, if model works
#i'll comment this out
#X_test_scaled = pd.read_csv("30_testdatascaled.csv")
#print(model.predict(X_test_scaled))