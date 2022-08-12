import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
#from pytorch_pretrained_bert import BertTokenizer, BertConfig
#from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from transformers import BertTokenizer, AdamW, BertConfig, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import time

from utils import (get_data, 
                   label_summary,
                   label_encoding,
                   train_test_splitting,
                   make_tokenizer,
                   BertModel,
                   flat_accuracy,
                   format_time)

class BertTrainer(object):
    def __init__(self, cfg_path) -> None:
        super().__init__()

        ## Load a Config File
        with open(cfg_path, 'r') as cfg:
            self.config = json.load(cfg)

        ## Load a Data
        self.data = get_data(self.config['main_parameters']['data_dir'])

        ## Label Summary
        label_summary(self.data, self.config['main_parameters']['label_column'])

        ## Label Encoding
        self.label_encoder, self.data[self.config['main_parameters']['label_column']] = label_encoding(self.data, label_column = self.config['main_parameters']['label_column'])

        ## Train, Validation Split
        self.train_docs, self.valid_docs, self.train_labels, self.valid_labels = train_test_splitting(self.data, 
                document_column = self.config['main_parameters']['document_column'], 
                label_column = self.config['main_parameters']['label_column'], 
                test_size = self.config['train_parameters']['test_size'], 
                random_seed = self.config['train_parameters']['random_seed'])


        ## Environment Setting
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        print("We can use GPU: {} \n".format(torch.cuda.get_device_name(0)))


        ## Bert Tokenizer
        self.train_dataset = make_tokenizer(self.train_docs, 
                                            self.train_labels, 
                                            pretrained_version = self.config['train_parameters']['pretrained_version'], 
                                            max_len = self.config['train_parameters']['max_len'])
        self.valid_dataset = make_tokenizer(self.valid_docs, 
                                            self.valid_labels, 
                                            pretrained_version = self.config['train_parameters']['pretrained_version'], 
                                            max_len = self.config['train_parameters']['max_len'])

        ## Make torch dataset
        # We'll take training samples in random order. 
        self.train_dataloader = DataLoader(
                    self.train_dataset,  # The training samples.
                    sampler = RandomSampler(self.train_dataset), # Select batches randomly
                    batch_size = self.config['train_parameters']['batch_size'] # Trains with this batch size.
                )

        # For validation the order doesn't matter, so we'll just read them sequentially.
        self.valid_dataloader = DataLoader(
                    self.valid_dataset, # The validation samples.
                    sampler = SequentialSampler(self.valid_dataset), # Pull out batches sequentially.
                    batch_size = self.config['train_parameters']['batch_size'] # Evaluate with this batch size.
                )        

        ## Load pretrained BERT model
        self.model = BertModel(pretrained_version = self.config['train_parameters']['pretrained_version'], 
                               class_number = len(self.label_encoder.classes_))
        self.model.cuda()


        ## Optimizer definition
        self.params = list(self.model.named_parameters())
        self.optimizer = AdamW(self.model.parameters(),
                               lr = self.config['train_parameters']['learning_rate'], 
                               eps = self.config['train_parameters']['adam_epsilon'] 
                               )

        ## Schedular definition
        # Total number of training steps is [number of batches] x [number of epochs]. 
        # (Note that this is not the same as the number of training samples).
        total_steps = len(self.train_dataloader) * self.config['train_parameters']['epochs']

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                    num_warmup_steps = 0, # Default value in run_glue.py
                                                    num_training_steps = total_steps)

        ## Set Seed (Reproductability)
        random.seed(self.config['train_parameters']['random_seed'])
        np.random.seed(self.config['train_parameters']['random_seed'])
        torch.manual_seed(self.config['train_parameters']['random_seed'])
        torch.cuda.manual_seed_all(self.config['train_parameters']['random_seed'])

        ## Training
        # We'll store a number of quantities such as training and validation loss, 
        # validation accuracy, and timings.
        training_stats = []

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        # For each epoch...
        for epoch_i in range(0, self.config['train_parameters']['epochs']):
    
            # ========================================
            #               Training
            # ========================================
    
            # Perform one full pass over the training set.

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.config['train_parameters']['epochs']))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_loss = 0

            # Put the model into training mode. Don't be mislead--the call to 
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            self.model.train()

            # For each batch of training data...
            for step, batch in enumerate(self.train_dataloader):

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)
            
                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(self.train_dataloader), elapsed))

                # Unpack this training batch from our dataloader. 
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the 
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids 
                #   [1]: attention masks
                #   [2]: labels 
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because 
                # accumulating the gradients is "convenient while training RNNs". 
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                self.model.zero_grad()        

                # Perform a forward pass (evaluate the model on this training batch).
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # It returns different numbers of parameters depending on what arguments
                # arge given and what flags are set. For our useage here, it returns
                # the loss (because we provided labels) and the "logits"--the model
                # outputs prior to activation.
                loss, logits = self.model(b_input_ids, 
                                          token_type_ids=None, 
                                          attention_mask=b_input_mask, 
                                          labels=b_labels,
                                          return_dict=False)

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value 
                # from the tensor.
                total_train_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                self.optimizer.step()

                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(self.train_dataloader)            
    
            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))
        
            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            self.model.eval()

            # Tracking variables 
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in self.valid_dataloader:
        
                # Unpack this training batch from our dataloader. 
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using 
                # the `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids 
                #   [1]: attention masks
                #   [2]: labels 
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
        
                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():        

                    # Forward pass, calculate logit predictions.
                    # token_type_ids is the same as the "segment ids", which 
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    # The documentation for this `model` function is here: 
                    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                    # Get the "logits" output by the model. The "logits" are the output
                    # values prior to applying an activation function like the softmax.
                    (loss, logits) = self.model(b_input_ids, 
                                                token_type_ids=None, 
                                                attention_mask=b_input_mask,
                                                labels=b_labels,
                                                return_dict=False)
            
                # Accumulate the validation loss.
                total_eval_loss += loss

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                total_eval_accuracy += flat_accuracy(logits, label_ids)
        

            # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(self.valid_dataloader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(self.valid_dataloader)
    
            # Measure how long the validation run took.
            validation_time = format_time(time.time() - t0)
    
            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))






if __name__ == "__main__":
    sc = BertTrainer("./bert_config.params")
    x = 1



    