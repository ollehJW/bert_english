import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def get_data(data_dir, sheet_name = None):
    """
    - Role
        Load a data.

    - Inputs:
        data_dir: Data directory.
        sheet_name: Name of sheet (if exist).

    - Outputs:
        data: Dataframe.
    """
    print("----- Data Loading -----")
    if data_dir.split('.')[-1] in ['txt', 'csv']:
        data = pd.read_csv(data_dir)
    elif data_dir.split('.')[-1] in ['xlsx', 'xls']:
        if sheet_name == None:
            data = pd.read_excel(data_dir)
        else:
            data = pd.read_excel(data_dir, sheet_name=sheet_name)

    print("Data is loaded!")
    print("Rows: {}, Columns: {} \n".format(data.shape[0], data.shape[1]))

    return data

def label_summary(data, label_column = 'class'):
    """
    - Role
        Print statistical summary for labels.

    - Inputs:
        data: Dataframe.
        label_column: Column with the label.

    - Outputs:
        None
    """
    print('----- Label summary -----')
    print('There are {} unique labels.'.format(len(set(data[label_column]))))
    label_counter = Counter(data[label_column])
    for (label, Count) in zip(label_counter.keys(), label_counter.values()):
        print("{}: {}".format(label, Count))

    print('\n')


def label_encoding(data, label_column = 'class'):
    """
    - Role
        Encodes labels to integer.

    - Inputs:
        data: Dataframe.
        label_column: Column with the label.

    - Outputs:
        label_encoder: Label Encoder
        encoded_labels: Encoded labels
    """
    print('----- Label Encoding -----')
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(data[label_column])
    for i in range(len(label_encoder.classes_)):
        print("{}: {}".format(i, label_encoder.classes_[i]))
    
    print("\n")

    return label_encoder, encoded_labels


def train_test_splitting(data, document_column = 'document', label_column = 'class', test_size = 0.2, random_seed = 1004):
    """
    - Role
        Split data into train and test samples.

    - Inputs:
        data: Dataframe.
        document_column: Column with the document.
        label_column: Column with the label.
        test_size: Proportion of test samples in splitting.
        random_seed: Random seed of splitting.

    - Outputs:
        train_docs: Documents of train samples.
        train_labels: Labels of train samples.
        test_docs: Documents of test samples.
        test_labels: Labels of tEst samples.
    """

    train_docs, test_docs, train_labels, test_labels = train_test_split(data[document_column], data[label_column], 
                    test_size=test_size, random_state=random_seed, stratify = data[label_column])
    
    return train_docs, test_docs, train_labels, test_labels


