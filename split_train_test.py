import numpy as np
import pandas as pd

def print_dataset_details(dataset, common_user_ids):
    ''' This function prints the detail of dataset '''
    counts = []
    for user in common_user_ids:
        common_pos = dataset.query('(expected_result == 1) and (user_id in @user)')['user_id']
        common_neg = dataset.query('(expected_result != 1) and (user_id in @user)')
        counts.append((common_pos.shape[0], common_pos.shape[0]))
        counts.sort(reverse = True)
    print("\nGENUINE & FRAUD RECORD COUNTS VIEW : \n\n",counts)

def print_train_test_value_counts(train_df, test_df, train_user_ids ,test_user_ids):
    train_value_counts = train_df.query('user_id in @train_user_ids').user_id.value_counts()
    test_value_counts = test_df.query('user_id in @test_user_ids').user_id.value_counts()
    print("\n\nTRAIN :\n\n", train_value_counts)
    print("\n\nTEST :\n\n", test_value_counts)
    
    
def id_split_train_test(dataset, max_range_of_records, min_range_of_records):
    ''' This function decides the 'user_id' role for train & test '''
    # Analyse user and there fraud & genuine records
    all_user_ids = dataset.user_id.value_counts().keys().tolist()
    pos = dataset.query('(expected_result == 1) and (user_id in @all_user_ids)')
    neg = dataset.query('(expected_result != 1) and (user_id in @all_user_ids)')

    # Split user's into train and test set
    test_user_ids = []
    train_user_ids = []
    drop_user_ids = []
    
    common_users_in_pos_neg = set(pos.user_id.value_counts().keys()) & set(neg.user_id.value_counts().keys())
    
    common_pos = dataset.query('(expected_result == 1) and (user_id in @common_users_in_pos_neg)')
    common_neg = dataset.query('(expected_result != 1) and (user_id in @common_users_in_pos_neg)')
    common_pos_count = common_pos.user_id.value_counts().sort_index()    
    common_neg_count = common_neg.user_id.value_counts().sort_index()
    
#     print("\n---  VALIDATE RECEIVED INPUTS  ---")
#     print(len(common_pos.user_id.unique()) ,  len(common_neg.user_id.unique()))
#     print(len(common_pos_count.keys()) ,  len(common_neg_count.keys()))
    
    for ind in range(len(common_users_in_pos_neg)):
        
        if common_pos_count.values[ind]>= max_range_of_records and common_neg_count.values[ind]>= max_range_of_records:
            train_user_ids.append(common_pos_count.keys()[ind])
            
        elif((common_pos_count.values[ind]< max_range_of_records and common_pos_count.values[ind]>= min_range_of_records) and (common_neg_count.values[ind]< max_range_of_records and common_neg_count.values[ind]>= min_range_of_records)):
            test_user_ids.append(common_pos_count.keys()[ind])
            
        else:
            drop_user_ids.append(common_pos_count.keys()[ind])
           
    # dropped users        
    drop_user_df = dataset.query('(user_id in @drop_user_ids)')
    drop_count = drop_user_df.user_id.value_counts()

    print("\nTotal Unique USER IDS in dataset : ",len(all_user_ids))
    print("\n**** USERS SELECTION IS BASED ON EQUAL NUMBER OF GENUINE & FRAUD RECORDS ****")
    print("\nTotal Unique USER IDS having both fraud & genuine records : ",len(common_users_in_pos_neg))
    print(f"\nUsers in TRAIN :  {len(train_user_ids)}")
    print(f"\nUsers in TEST: {len(test_user_ids)}")
    print(f"\nUsers DROPPED: {len(drop_user_ids)}")
#     print(f"drop count : \n{drop_count}")
    
    return common_users_in_pos_neg, train_user_ids ,test_user_ids


def get_train_test_split_data(dataset, train_user_ids , test_user_ids, number_of_train_records, number_of_test_records):     
    
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    
    # TRAIN DATASET
    
    for user in train_user_ids:
        train_df = train_df.append(dataset.query('(user_id in @user) and (expected_result == 1)')[:number_of_train_records])
        train_df = train_df.append(dataset.query('(user_id in @user) and (expected_result == -1)')[:number_of_train_records])
    train_df = train_df.sort_values(by=['user_id'])
        
    # TEST DATASET
        
    for user in test_user_ids:
        test_df = test_df.append(dataset.query('(user_id in @user) and (expected_result == 1)')[:number_of_test_records])
        test_df = test_df.append(dataset.query('(user_id in @user) and (expected_result == -1)')[:number_of_test_records])
    test_df = test_df.sort_values(by=['user_id'])
        
        
    print("\n**** DATASETS VALUE ****\n") 
    print("TRAIN DF : ", train_df.shape)
    print("TEST DF : ", test_df.shape,"\n")
        
    return train_df, test_df