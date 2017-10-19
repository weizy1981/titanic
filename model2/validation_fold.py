from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def validation_model(model, X, y):
    seed = 7
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

    results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    print('Accuracy: %.2f%% (%.2f)' % (results.mean() * 100, results.std()))