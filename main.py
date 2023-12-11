import pickle
import numpy as np
import copy
import torch
import argparse

from utils import *
from models import *
from ssl_tools import *

if __name__ == "__main__" : 
    
    parser = argparse.ArgumentParser('Proposed Method.')
    parser.add_argument('-data', '--data', type=str, default='temporal_mag')        
    parser.add_argument('-task', '--task', type=str, default='task1')        
    parser.add_argument('-device', '--device', type=str, default='cuda:0')
    parser.add_argument('-epoch', '--epoch', type=int, default=10)
    parser.add_argument('-lr', '--lr', type=float, default=0.001)
    parser.add_argument('-mask_rate', '--mask_rate', type=float, default=0.3)
    args = parser.parse_args()
    
    device = args.device
    task = args.task
    data = args.data
    epoch = args.epoch
    lr = args.lr
    mask_rate = args.mask_rate
    
    if data not in ["temporal_mag", "temporal_dblp"] :     
        raise TypeError("Dataset name is wrong")
    
    if task == "task1" : 
        with open('{0}/{0}_split_E.pickle'.format(data), 'rb') as f : 
            each_HE = pickle.load(f)

        with open('{0}/{0}_split_X.pickle'.format(data), 'rb') as f : 
            each_X = pickle.load(f)
            
        with open('{0}/{0}_split_index.pickle'.format(data), 'rb') as f : 
            each_IDX = pickle.load(f)
            
        with open('{0}/{0}_split_pair_match.pickle'.format(data), 'rb') as f : 
            match_time = pickle.load(f)
            
        with open('{0}/{0}_split_match.pickle'.format(data), 'rb') as f : 
            each_match = pickle.load(f)
            
        total_seed = 5

        PERF = np.zeros((5, 10))
        
        for i in range(total_seed) : 

            loader = Task1DataLoader(match = each_match, 
                         entireHE = each_HE, 
                         init_seed = i, device = device, 
                         ratio = 0.05)
                  
            loader.split_train_valid_test(i)
                
            train_ind1, train_ind2, train_label = loader.load_dataset("train")
            valid_ind1, valid_ind2, valid_label = loader.load_dataset("valid")
            test_ind1, test_ind2, test_label = loader.load_dataset("test")
                
            train_triple = reindex_indices(train_ind1, train_ind2)
            valid_triple = reindex_indices(valid_ind1, valid_ind2)
            test_triple = reindex_indices(test_ind1, test_ind2)
            
            if data == 'mag' : 
                ranger = np.arange(111, 121)
                dim = 64
            else : 
                ranger = np.arange(27,37)
                dim = 128
                  
            ### Pretraining SSL
                  
            encoder = HyperEncoder(in_dim = each_X[0].shape[1], 
                                   edge_dim = dim, node_dim = dim, num_layers=2).to(device)

            decoder = HyperDecoder(in_dim = dim, 
                                   edge_dim = each_X[0].shape[1], node_dim = each_X[0].shape[1], num_layers=2).to(device)

            trainer = HyperMAETrainer(Xs = each_X, Es = each_HE, IDXs = each_IDX, 
                                      encoder = encoder, decoder = decoder, device = device, do_e_aug = True)
                  
            parameter, l_lists = trainer.fit(epoch = epoch, lr = lr,
                        drop_feature=mask_rate, gamma=1, save_model=True, explicit_bound = ranger)
            
            for ik, time in enumerate(ranger) :
                
                encoder = HyperEncoder(in_dim = each_X[0].shape[1], 
                           edge_dim = dim, node_dim = dim, num_layers=2).to(device)
                
                save_param = parameter[ik][0]

                encoder.load_state_dict(save_param)
                each_Z = createZ(each_X = each_X, each_HE = each_HE, last_time = time, encoder = encoder, 
                        device = device, feature_type = "LSTM", time_mapper = match_time)
                torch.cuda.empty_cache()
                classifier = TaskLSTM(in_dim = dim, hidden_dim = dim, drop_p = 0.5).to(device)
                cur_max = train_LSTM_task1(model = classifier, Xs = each_Z, loader = loader, 
                                      train_triple = train_triple, valid_triple = valid_triple, 
                                      test_triple = test_triple, lr = 0.001, epochs = 100, device = device, target_time = time)
                  
                print(i, time, cur_max)
                PERF[i, ik] = cur_max
        print(np.mean(PERF, 0))
                
    else :
    
        with open('{0}/{0}_orig_E.pickle'.format(data), 'rb') as f : 
            each_HE = pickle.load(f)

        with open('{0}/{0}_orig_X.pickle'.format(data), 'rb') as f : 
            each_X = pickle.load(f)
                  
        with open('{0}/{0}_orig_Y.pickle'.format(data), 'rb') as f : 
            each_Y = pickle.load(f)
            
        with open('{0}/{0}_orig_index.pickle'.format(data), 'rb') as f : 
            each_IDX = pickle.load(f)
            
        with open('{0}/{0}_orig_pair_match.pickle'.format(data), 'rb') as f : 
            match_time = pickle.load(f)
            
        print("Data Loading Done")
        
        total_seed = 5
        batch_cases = 10
        batch_size = 100000
        eval_batch_size = 200000

        PERF = np.zeros((5, 10))

        for i in range(total_seed) : 
            
            loader =  Task2DataLoader(total_label = each_Y, init_seed = i, 
                                device = device, sampling=True, split_ratio=0.05,
                                 n_train_per_each=50000, n_valid_per_each=50000, n_test_per_each=50000)

            loader.device = device

            totalI1, totalI2, totalVs = [], [], [] # Training batches
            ValidB = loader.partial_pair_loader(batch_size = 100000, seed = i, device='cpu', mode='valid')
            TestB = loader.partial_pair_loader(batch_size = 1000000, seed = i, device='cpu', mode='test')

            for tmp_seed in range(batch_cases) : 

                I1, I2, Vs = create_batch_pairs(positive_pairs = loader.train_positive_pair, negative_pairs = loader.train_negative_pair, 
                                   valid_Vs1 = None, valid_Vs2 = None, valid_labels = None, target_time = 0, seed = tmp_seed,
                                   batch_size = batch_size, device = device, data_type = 'train')

                totalI1.append(I1)
                totalI2.append(I2)
                totalVs.append(Vs)

            if data == 'mag' : 
                ranger = np.arange(111, 121)
                dim = 64
            else : 
                #ranger = np.arange(1,10)
                ranger = np.arange(27,37)
                dim = 128
                  
            encoder = HyperEncoder(in_dim = each_X[0].shape[1], 
                                   edge_dim = dim, node_dim = dim, num_layers=2).to(device)

            decoder = HyperDecoder(in_dim = dim, 
                                   edge_dim = each_X[0].shape[1], node_dim = each_X[0].shape[1], num_layers=2).to(device)

            trainer = HyperMAETrainer(Xs = each_X, Es = each_HE, IDXs = each_IDX, 
                                      encoder = encoder, decoder = decoder, device = device, do_e_aug = True)
                  
            parameter, l_lists = trainer.fit(epoch = epoch, lr = lr,
                        drop_feature=mask_rate, gamma=1, save_model=True, explicit_bound = ranger)
                
            for ik, time in enumerate(ranger)  : 

                cur_goat = 0

                V1, V2, VD, VY = create_batch_pairs(positive_pairs = None, negative_pairs = None, 
                               valid_Vs1 = ValidB[0], valid_Vs2 = ValidB[1], valid_labels = ValidB[2], target_time = time,
                               batch_size = eval_batch_size, device = device, data_type = 'valid', seed = i)

                T1, T2, TD, TY = create_batch_pairs(positive_pairs = None, negative_pairs = None, 
                               valid_Vs1 = TestB[0], valid_Vs2 = TestB[1], valid_labels = TestB[2], target_time = time,
                               batch_size = eval_batch_size, device = device, data_type = 'test', seed = i)
                    
                save_param = parameter[ik][0] # Originally 4
                encoder = HyperEncoder(in_dim = each_X[0].shape[1], 
                           edge_dim = dim, node_dim = dim, num_layers=2).to(device)
                encoder.load_state_dict(save_param)
                each_Z = createZ(each_X = each_X, each_HE = each_HE, last_time = time, encoder = encoder, 
                            device = device, feature_type = "LSTM", time_mapper = match_time)
                torch.cuda.empty_cache()

                classifier = TaskLSTM(in_dim = dim, hidden_dim = dim, drop_p = 0.5).to(device)

                cur_max = train_LSTM_task2(model = classifier, Xs = each_Z, loader = loader, evaluation_time = time, 
                                          lr = 0.001, epochs= 100, device = device, 
                                          batch_size = batch_size, I1 = totalI1, I2 = totalI2, Vs = totalVs, 
                                          V1 = V1, V2 = V2, VV = VD, VY = VY, 
                                          T1 = T1, T2 = T2, TV = TD, TY = TY)

                print(i, time, cur_max)
                PERF[i, ik] = cur_max
        print(np.mean(PERF, 0))