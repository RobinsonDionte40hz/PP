import numpy as np

def calculate_rmse(predictions, actuals):
    """
    Calculate Root Mean Square Error between predictions and actual values.
    
    Parameters:
    -----------
    predictions : array-like
        Predicted values
    actuals : array-like
        Actual experimental values
        
    Returns:
    --------
    float
        RMSE value
    """
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    if len(predictions) != len(actuals):
        raise ValueError("Predictions and actuals must have same length")
        
    return np.sqrt(np.mean((predictions - actuals) ** 2))

def calculate_normalized_rmse(predictions, actuals):
    """
    Calculate normalized RMSE (as percentage of mean actual value).
    
    Parameters:
    -----------
    predictions : array-like
        Predicted values
    actuals : array-like
        Actual experimental values
        
    Returns:
    --------
    float
        Normalized RMSE as percentage
    """
    rmse = calculate_rmse(predictions, actuals)
    mean_actual = np.mean(np.abs(actuals))
    
    if mean_actual == 0:
        return float('inf')
        
    return (rmse / mean_actual) * 100
