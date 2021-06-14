from sklearn.pipeline import Pipelinefrom sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from om_data_challenge.data import om_get_data, om_clean_text


c
X= french_train_1['clean_text']
y= french_train_1['label']lass Trainer():
    def __init__():
        self.X = X
        self.y = y


    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        X= french_train_1['clean_text']
        y= french_train_1['label']

        # Create Pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('nb', MultinomialNB()),
        ])

        # Set parameters to search
        parameters = {
            'tfidf__ngram_range': ((1,1), (2,2), (1,2), (2,3), (3,3)),
            'nb__alpha': (0.1, 1, 10),}

        # Perform grid search
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, 
                                verbose=1, scoring = "accuracy", 
                                refit=True, cv=10)

        grid_search.fit(X,y)



