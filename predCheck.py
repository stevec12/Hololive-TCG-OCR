import os
from pathlib import Path
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

# Seed NumPy randomness if reproducability desired
np.random.seed(4444)
predParentDir = Path.cwd() / "Preds"

def numsOnly(pred : str):
    # Simply removes non-numeric characters
    if pd.isna(pred):
        return np.nan
    result = ''.join(filter(str.isdigit, pred))
    # Avoid typecasting error of the empty string
    return int(result) if result != '' else np.nan
    
def contextChecks(pred: str):
    # Applies prediction corrections relevant to the given context
    # Here, the context is a dollar value of a trading card
    
    # Ensure there are some numbers in the prediction
    if pd.isna(numsOnly(pred)):
        return np.nan
    corrected = delNonsenseChars(pred)
    corrected = numericCommaContext(pred)
    return corrected

def delNonsenseChars(pred: str) -> str:
    """
    This function removes non-digits throughout the string, 
    except for those that make sense in the context. 
        
    Parameters
    ----------
    pred : str
        non-empty prediction string with some digits.

    Returns
    -------
    str
        cleaned up prediction string.
    """
    # For the card values, only commas would be useful for contextualizing the 
    #   prediction later on
    sensical_nondigits = [',']
    adj_str = []
    for c in pred:
        if c.isdigit() or c in sensical_nondigits:
            adj_str.append(c)
    return ''.join(adj_str)

def numericCommaContext(pred: str) -> int:
    """
    Uses the context of commas to adjust the prediction to be more reasonable,
        by filling in sampled digits for missing characters
    
    Parameters
    ----------
    pred : str
        non-empty prediction string made up of digits and commas.

    Returns
    -------
    int
        adjusted prediction integer determined by context.

    """
    later_comma = 0 # Flag on whether the 1st comma has been found
    consecutive_digits = 0 # Count the consecutive digits until a comma
    final_pred = "" # Store the final_prediction
    
    # 1st: check whether the leading character is a comma or 0
    if pred[0] == 0 or pred[0] == ',':
        # If so, sample 1 digit from 1-9
        final_pred += str(np.random.randint(low=1,high=10))
        if pred[0] == ',':
            later_comma = 1
    else:
        final_pred += pred[0]
            
    # 2nd: check remaining characters
    for c in pred[1:]:
        if c == ',':
            # Later commas should have at least 3 digits between them
            if later_comma and consecutive_digits < 3:
                # Fill remaining digits with sampled digits from 0-9
                for i in range(3-consecutive_digits):
                    final_pred += str(np.random.randint(low=0,high=10))
            else: # Initial comma can have any number of digits before it
                later_comma = 1
            consecutive_digits = 0
        else:
            # If just a digit, simply append and add to the consecutive_digits
            consecutive_digits += 1
            final_pred += c
    # 3rd: When ending with < 3 consective digits after a comma has been found
    if later_comma and consecutive_digits < 3:
        for i in range(3-consecutive_digits):
            final_pred += str(np.random.randint(low=0,high=10))
    return final_pred

def predVoting(preds_ssID: pd.DataFrame, preds: pd.DataFrame) -> pd.Series:
    """
    Votes between screenshots taken in a similar timeframe to form a singular 
        answer, adding this as a column to the original DataFrame. 
    Remark: Could use a similarity score to prior images instead of using 
        similar timeframes, if calculated earlier. 

    Parameters
    ----------
    preds_ssID: 
        ssIDs of the predictions
    preds : pd.DataFrame
        Adjusted predictions, sorted in ascending order of ssID

    Returns
    -------
    pd.DataFrame
        Original predictions with a new 

    """
    start = 0
    end = 1
    frame_gap = 300

    pred_votes = {preds[start]:1} # {pred : #votes} pairs
    while end < len(preds_ssID): 
        # If the difference between the ssID frame numbers is less than the
        #   frame gap, consider them in the same timeframe
        # Note: preds_ssID[end][:-4] used to remove the '.jpg' suffix
        if int(preds_ssID[end][:-4]) - int(preds_ssID[start][:-4]) < frame_gap:
            pred_votes[preds[end]] = pred_votes.setdefault(preds[end],0) + 1
            end += 1
        else:
            # Vote on last set of samey ssIDs
            max_votes = max(pred_votes.values())
            voted = [] # Store preds with the most votes
            for pred in pred_votes.keys():
                if pred_votes[pred] == max_votes:
                    voted.append(pred)
            final_pred = 0
            if pd.isna(voted[0]): final_pred = np.nan
            else:
                none_ctr = 0
                for v in voted: # Don't add None
                    if not pd.isna(v): final_pred += int(v) 
                    else: none_ctr = 1

                # Take average of the top voted results
                final_pred = int(final_pred/(len(voted)-none_ctr))

            # Assign value to first occurence, NaN to remaining occurences
            preds.iloc[start] = final_pred
            for i in range(start+1, end):
                preds.iloc[i] = np.nan
            # Close the interval
            pred_votes = {preds[end]:1}
            start = end
            end += 1
    # Clean up the last set of same timeframe images
    if start+1 < len(preds):
        # Vote on last set of samey ssIDs
        max_votes = max(pred_votes.values())
        voted = [] # Store preds with the most votes
        for pred in pred_votes.keys():
            if pred_votes[pred] == max_votes:
                voted.append(pred)
        final_pred = 0
        if pd.isna(voted[0]): final_pred = np.nan
        else:
            none_ctr = 0
            for v in voted: # Don't add None
                if not pd.isna(v): final_pred += int(v) 
                else: none_ctr = 1

            # Take average of the top voted results
            final_pred = int(final_pred/(len(voted)-none_ctr))

        # Assign value to first occurence, NaN to remaining occurences
        preds.iloc[start] = final_pred
        for i in range(start+1, end):
            preds.iloc[i] = np.nan

    return preds



def predCheck(mem_name: str):
    # Do some contextual adjustments and inference of values from the predictions
    predDir = predParentDir / mem_name
    vidPreds = [f for f in os.listdir(predDir)]
    
    for pred in vidPreds:
        predPath = predDir / pred
        pred_df = pd.read_csv(predPath, sep=" ", index_col=None, dtype=(str,str))
        # Add new column that simply strips 
        pred_df["Numeric Pred"] = pred_df["Prediction"].apply(numsOnly)
        # Add a column of the context adjusted number
        pred_df["Pred Adjusted"] = pred_df["Prediction"].apply(contextChecks)
        # Find images that are taken at similar times, thus likely to be the same card. 
        # Then vote for the most likely text, placing this in place of the first occurence
        #   and replacing the remainder with NaN.
        # Here, ssIDs of similar values can be grouped.
        pred_df["Pred Grouped"] = pred_df["Pred Adjusted"]
        pred_df["Pred Grouped"] = predVoting(pred_df["ssID"], pred_df["Pred Adjusted"])
        
        # Write to a file
        adj_pred_df = pred_df[["ssID", "Pred Grouped"]].dropna(axis=0)
        
        # Strip "..._initial_preds.txt" from the file names to add "_adj_preds.txt"
        adjPredPath = predDir / (pred[:-18]+"_adj_preds.txt")
        adj_pred_df.to_csv(adjPredPath, sep=' ', index=False, )
    
            
        

predCheck("Usada Pekora")