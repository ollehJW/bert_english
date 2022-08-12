from transformers import BertTokenizer
import torch
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification, AdamW, BertConfig

def make_tokenizer(sentences, labels, pretrained_version = "bert-base-uncased", max_len = 128):
    """
    - Role
        Make tokenized TensorDataset.

    - Inputs:
        sentences: Documents.
        labels: Labels.
        pretrained_version: Pretrain version of BERT tokenizer.
        max_len: Max length for tokenizing per setences.

    - Outputs:
        dataset: TensorDataset.
    """
    # Load the BERT tokenizer.
    print('----- Loading BERT tokenizer -----')
    tokenizer = BertTokenizer.from_pretrained(pretrained_version, do_lower_case=True)
    print('----- Successfully Loaded ----- \n')

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    print('----- Tokenizing our datasets -----')
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = max_len,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',  # Return pytorch tensors.
                        truncation = True,    
                   )
    
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
    
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    print('----- Successfully Tokenized ----- \n')
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(list(labels))

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    return dataset


def BertModel(pretrained_version = "bert-base-uncased", class_number = 5):
    """
    - Role
        Make tokenized TensorDataset.

    - Inputs:
        pretrained_version: Pretrain version of BERT model.
        class_number: Unique class number.

    - Outputs:
        model: Pretrained BERT model.
    """

    print('----- Load Pretrained BERT Model -----')
    # Load BertForSequenceClassification, the pretrained BERT model with a single 
    # linear classification layer on top. 
    model = BertForSequenceClassification.from_pretrained(
        pretrained_version,
        num_labels = class_number, 
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )
    print('----- Successfully Loading BERT Model ----- \n')
    return model


