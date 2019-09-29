import numpy as np
import pandas as pd

# remove rows having null values in the sensor part
value_to_test = 0.0
col_start = 2
col_end = 23

def load_dataset(dataset_path):
    ''' This function loads the dataset '''
    dataset = pd.read_csv(dataset_path,low_memory=False)
    print(f"\nOriginal Shape : {dataset.shape}")
    return dataset

def select_required_features(dataset, keywords):
    ''' This function selects the keyword based columns from dataset '''
    required_cols = []           
    for col in dataset.columns:
        for keyword in keywords:
            if (keyword in col):
                if (keyword == 'mean'):
                    if ('motion' in col):
                        required_cols.append(col)
                else:
                    required_cols.append(col)    
    # dataset with selected cols
    dataset = dataset[required_cols]
    return dataset

def free_dataset_from_given_value(dataset, value_to_test, col_start, col_end):
    ''' This function takes dataset, value to check in columns, starting column, ending column till which the value to be find '''
    # store clean data  
    new_dataset = pd.DataFrame(columns=dataset.columns)
    # count of elimated rows
    rows_eliminated = 0
    # iterrows() return tuple of (row_index, row_values)
    for i in dataset.iterrows():
        col_count = 0
        # list to store col values
        l = []
        # store col values
        for col_value in i[1]:
            # check the mentioned column values 
            if col_count >= col_start and col_count < col_end :
                l.append(col_value)
            col_count+=1
#         print(l)
        # check the values in the single row's columns
        if all([value == value_to_test for value in l]):
#             print("List : ",i[0], l)
            rows_eliminated+=1
        else:
            new_dataset = new_dataset.append(i[1])
    print(f"\nAfter pre-processing : {new_dataset.shape}")
#     print(f"{'_'*100}")
    return new_dataset


def prepare_dataset(dataset_path, keywords):
    ''' This function preprocess the dataset and returns the clean & valid dataset'''
    # load dataset
    dataset = load_dataset(dataset_path)
    # drop 0 expected results
    dataset = dataset[dataset['expected_result'] != 0]
    # drop columns where all the sensors are unavailable
    dataset = dataset[dataset['sensors'] == 'YAMGNRa']
    # get mentioned columns only
    dataset = select_required_features(dataset, keywords) 
    # drop null device_uids
    dataset = dataset[dataset.user_id.notnull()]
    # drop continuous 0 valued columns
    dataset = free_dataset_from_given_value(dataset, value_to_test, col_start, col_end)
    # drop sensors column
    dataset = dataset.drop(columns='sensors')
    return dataset


def one_hot_encode(dataset, column_name):
    ''' This function takes a dataframe & column and returns the one-hot-encoded dataframe column '''
    uniques = dataset[column_name].unique()
    uniques_to_int = {unique: i  for i, unique in enumerate(uniques)}
    int_to_uniques = {i: unique for i, unique in enumerate(uniques)}
    dataset = dataset.replace(uniques_to_int)
    return dataset

def saimese_input_structure(dataset):
    ''' This function converts the dataset into siamese input structure '''
    new_dataset = pd.DataFrame()
    motion_list = list(np.array([np.array(val) for val in dataset.values[:,1:-1].tolist()]))
    #print([np.array(val) for val in dataset.values[:,1:-1].tolist()])
    new_dataset.insert(0, "sequence",motion_list)
    new_dataset.insert(1, "user_id", dataset.values[:,-1].tolist())
    return new_dataset
