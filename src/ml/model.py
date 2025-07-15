from sklearn.ensemble import RandomForestRegressor

def train_model(X, y):
    model = RandomForestRegressor(
        bootstrap= False, max_depth= None, max_features= 0.5, min_samples_leaf= 1, min_samples_split= 5, n_estimators= 200
    )
    model.fit(X, y)
    return model