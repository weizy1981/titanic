from sklearn.ensemble import RandomForestClassifier

def select_feature(X, y):
    model = RandomForestClassifier()
    fit = model.fit(X, y)
    print(fit.feature_importances_)