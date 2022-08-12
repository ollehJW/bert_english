import numpy as np
import time
from transformers import get_linear_schedule_with_warmup
import datetime

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    """
    - Role
        Calculate an accuracy metric.

    - Inputs:
        preds: Model output.
        labels: True label.

    - Outputs:
        Accuracy.
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    """
    - Role
        Takes a time in seconds and returns a string hh:mm:ss.
    """
    
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))