import numpy as np
from math import factorial

def delete_redundant_columns(input_array, index):
    ''' This function remove the element from array '''
    return np.delete(input_array, index, axis=1)
    
def get_positive_pairs(df, users):
    ''' This function CREATE POSITIVE PAIRS

    ** no repeatation in pairs (unique pair formation)
    - EXAMPLE :
        - l1 = [1, 2, 3, 4, 5]
        - pos_pairs = [(1,2), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5),(4,5)]
        - nCr = n! / [(n-r)! * r!]
    '''
    positive_pairs = []

    for user in users:
        user_data_df = df.query('(user_id == @user)')
        
        user_data = user_data_df.as_matrix()
        positive_df = user_data
        
        for index_1 in reversed(range(len(positive_df))):    
            # get left-side record
            record_1 = positive_df[index_1]
            # make pairs with all remaining records
            for index_2 in reversed(range(index_1)):
                # get right-side record
                record_2 = positive_df[index_2]    
                # make pairs of 2 records
                positive_pairs.append(np.hstack([record_1, record_2]))
                index_2 -= 1
            index_1 -= 1
      
    return np.vstack(positive_pairs)

def get_negative_pairs(negative_df, positive_df, users):
    ''' This function CREATE NEGATIVE PAIRS

    ** no repeatation in pairs (unique pair formation)
    - EXAMPLE :
        - l1 = [1, 2, 3, 4]
        - l2 = [5, 6, 7, 8]
        - pos_pairs = [(1,5), (1,6), (1,7), (1,8), (2,5), (2,6), (2,7), (2,8), (3,5),(3,6),(3,7),(3,8),(4,5),(4,6),(4,7),(4,8)] 
    - Pairs formed are unique
    - nCr = n! / (n-r)!
    '''
    negative_pairs = []
   
    for user in users:
       
        negative_user_data_df = negative_df.query('(user_id == @user)')
        positive_user_data_df = positive_df.query('(user_id == @user)')
        
        user_positive_data = positive_user_data_df.as_matrix()
        user_negative_data = negative_user_data_df.as_matrix()
    
        for index_1 in reversed(range(len(user_positive_data))):
            
            # get left-side record
            record_1 = user_positive_data[index_1]
            
            # make pairs with all remaining records
            for index_2 in reversed(range(len(user_positive_data))):
                # get right-side record
                record_2 = user_negative_data[index_2]    
                # make pairs of 2 records
                negative_pairs.append(np.hstack([record_1, record_2]))
                index_2 -= 1
            index_1 -= 1

    return np.vstack(negative_pairs)

    
    
def validate_positive_pairs(input_array_len, pairs_length):
    ''' This function validates the positive pairs formed '''
    validated_pairs_array_len =  int(factorial(input_array_len)/((factorial(input_array_len - 2))*(factorial(2))))
    print(validated_pairs_array_len, pairs_length)
    if validated_pairs_array_len == pairs_length:
        validation_flag = True
    else:
        validation_flag = False


def validate_negative_pairs(input_array_len, pairs_length):
    ''' This function validates the negative pairs formed '''
    validated_pairs_array_len =  input_array_len**2
    
    if validated_pairs_array_len == pairs_length:
        print("CORRECT NEGATIVE PAIRS FORMATION !")
    else:
        print("INCORRECT NEGATIVE PAIRS FORMATION !")


def balance_negative_pairs_equal_to_positive_pairs(negative_pairs):
    ''' This function balances the negative pair data and make it equal to positive pair data '''
    balanced_negative_pairs = None
    return balanced_negative_pairs

def validate_positive_and_negative_pairs(train_positive_pairs, train_negative_pairs):
    ''' This function validates the balancing of positive & negative data '''
    if train_positive_pairs == train_negative_pairs:
        print("\nBALANCED(equal) POSITIVE & NEGATIVE TRAINING DATA !")
        