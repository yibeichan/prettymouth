import numpy as np
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from himalaya.scoring import correlation_score, correlation_score_split

def reshape_y(y):
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    else:
        y = y
    return y

def get_ridgereg_tscv(df, emb_cols, predictor_cols):
    """
    Perform ridge regression with TimeSeriesSplit cross-validation.
    Parameters:
        df: pd.DataFrame
            The dataframe containing the data.
        emb_cols: np.array
            The array containing the embeddings.
        rating_type: str or a list of str
            The type of rating to predict.
    """
    # Split the data into train and test sets
    X = emb_cols.values
    y = df[predictor_cols].values
    
    # Define a list of alpha values for Ridge regression
    alphas = np.logspace(0, 10, 10)

    # Create a TimeSeriesSplit object for both inner and outer loops
    inner_tscv = TimeSeriesSplit(n_splits=5)
    outer_tscv = TimeSeriesSplit(n_splits=3)

    # Perform outer cross-validation
    outer_scores = []
    r2_scores = []
    adj_r2_scores = []
    for outer_train_index, outer_test_index in outer_tscv.split(X):
        X_outer_train, X_outer_test = X[outer_train_index, :], X[outer_test_index, :]
        y_outer_train, y_outer_test = y[outer_train_index], y[outer_test_index]
        # Scale the features
        scaler = StandardScaler()
        X_outer_train = scaler.fit_transform(X_outer_train)
        X_outer_test = scaler.transform(X_outer_test)
        
        # Perform inner cross-validation with RidgeCV
        ridge_cv = RidgeCV(alphas=alphas, cv=inner_tscv)
        ridge_cv.fit(X_outer_train, y_outer_train)
        
        best_alpha = ridge_cv.alpha_
        
        y_pred = ridge_cv.predict(X_outer_test)
        y_outer_test = reshape_y(y_outer_test)
        y_pred = reshape_y(y_pred)
        outer_score = correlation_score(y_outer_test, y_pred)
        outer_scores.append(outer_score)
        r2 = r2_score(y_outer_test, y_pred)
        r2_scores.append(r2)
        # calculate adjusted r2
        n = len(y_outer_test)
        p = X_outer_test.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        adj_r2_scores.append(adj_r2)
        print(f"Best alpha: {best_alpha:.3f}")
        print(f"Correlation score: {outer_score}")
        print(f"R2 score: {r2:.3f}")
        print(f"Adjusted R2 score: {adj_r2:.3f}")

    print(f"Average correlation score: {np.mean(outer_scores, axis=0)}")
    print(f"Average R2 score: {np.mean(r2_scores):.3f}")
    print(f"Average adjusted R2 score: {np.mean(adj_r2_scores):.3f}")
    return outer_scores, r2_scores, adj_r2_scores

def get_ridgereg_ttcv(df, emb_cols, predictor_cols):
    """
    This function runs ridge regression with train_test_split and cross-validation on the given data and returns the correlation, r2 and adjusted r2 scores
    Parameters:
    df: dataframe containing the data
    emb_cols: list of column names containing the embeddings
    rating_type: type of rating to be used for regression
    """
    # Split the data into train and test sets
    X = emb_cols
    y = df[predictor_cols].values
    # reshape y to be a 2D array if it is a 1D array
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    else:
        pass
    # Define a list of alpha values for Ridge regression
    alphas = np.logspace(0, 10, 10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    ridge_cv = RidgeCV(alphas=alphas, cv=5).fit(X_train, y_train)
    y_pred = ridge_cv.predict(X_test)
    corr = correlation_score_split(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    adjusted_r2 = 1 - (1-r2)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
    print(f"Correlation: {corr}", f"R2: {r2}", f"Adjusted R2: {adjusted_r2}")
    return corr, r2, adjusted_r2
