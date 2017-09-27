from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

def __turn_model(param_grid, model, X, y):
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold)
    results = grid.fit(X, y)
    print('Best: %f using %s' % (results.best_score_, results.best_params_))

def turn_RF(model, X, y):
    param_grid = {}
    param_grid['class_weight'] = [None, 'balanced', 'balanced_subsample']
    __turn_model(param_grid, model, X, y)

def turn_MLP(model, X, y):
    param_grid = {}
    param_grid['init'] = ['glorot_uniform', 'normal']
    __turn_model(param_grid, model, X, y)