import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from om_data_challenge.data import om_get_data, om_clean_text


class Trainer(object):
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y


    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        # Create Pipeline
        pipeline = Pipeline([
            ('vector', TfidfVectorizer()),
            ('model', MultinomialNB()),
        ])
        self.pipeline = pipeline

    def grid_search(self, X, y, pipeline):

        # Set parameters to search
        parameters = {
            'vector__ngram_range': ((1,1), (2,2), (1,2), (2,3), (3,3)),
            'model__alpha': (0.1, 1, 10),}

        # Perform grid search sur pipeline (metrics = accuracy)
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, 
                                verbose=1, scoring = "f1", 
                                refit=True, cv=10)

        grid_search.fit(X,y)
        return grid_search

    
    def run(self, X, y, pipeline):
        """
           set and train the pipeline 
           allow to run the model on X, y
        """
        self.pipeline.fit(X, y)
        return pipeline

    # Prediction on new tweet and evaluate metrics
    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = np.sqrt(((y_pred - y_true) ** 2).mean())
        return rmse

    # Predict
    def predict(self, pipeline, X_test):
        """
        return new prediction
        """
        y_pred = self.pipeline.predict(X_test)
        return y_pred
        
if __name__ == "__main__":

    df = om_get_data()
    df = om_clean_text(df)
    # set X and y
    y = df["label"]
    X = data['clean_text']
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # train
    my_trainer = Trainer(X_train,y_train)
    
    pipeline = my_trainer.set_pipeline()
    
    my_trainer.run(X_train, y_train, pipeline)

    # evaluate
    rmse = my_trainer.evaluate(X_test, y_test)
    prediction = my_trainer.predict(X_test)
    print(f"Sentiment is :" {prediction})
    print(f'rmse of the model is : {rmse}')