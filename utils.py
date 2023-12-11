
import pickle
import copy
import torch
import numpy as np

from tqdm import tqdm
from itertools import combinations, permutations
from sklearn.metrics import roc_auc_score, average_precision_score

## Task 1

class Task1DataLoader():

    def __init__(self, match, entireHE, init_seed, device, ratio):

        self.device = device
        self.seed = init_seed
        self.entire_pairs = copy.deepcopy(match)
        self.ratio = ratio
        self.opponent_bucket = [] ## Tells which node is an opponent of which node
        for i in range(len(match)) : 
            tmpdict = dict()
            for v in range(int(torch.max(entireHE[i][0]) + 1)) : 
                tmpdict[v] = -1
            self.opponent_bucket.append(tmpdict)
        
        for i in range(len(match)) : 
            idx1 = match[i][0]
            idx2 = match[i][1]
            for v1, v2 in zip(idx1, idx2) : 
                self.opponent_bucket[i][v1] = v2 # v1's original opponent is v2
                self.opponent_bucket[i][v2] = v1 # v2's original opponent is v2
                
        self.n_train = []
        for i in range(len(match)) : 
            curN = max(int(self.ratio * len(match[i][0])), 1)
            self.n_train.append(curN)
            
        #self.n_train = [int(self.ratio * len(match[i][0])) for i in range(len(match))]
        
    def split_train_valid_test(self, seed = 0) : 
        
        np.random.seed(seed) ## Fix seed for the current step
        
        self.train_pos_samples = [([0] * (self.n_train[i]), [0] * (self.n_train[i])) for i in range(len(self.opponent_bucket))]
        self.train_neg_samples = [([0] * (self.n_train[i]), [0] * (self.n_train[i])) for i in range(len(self.opponent_bucket))]
        
        self.valid_pos_samples = [([0] * (self.n_train[i]), [0] * (self.n_train[i])) for i in range(len(self.opponent_bucket))]
        self.valid_neg_samples = [([0] * (self.n_train[i]), [0] * (self.n_train[i])) for i in range(len(self.opponent_bucket))]
        
        self.test_pos_samples = []
        self.test_neg_samples = []
        
        node_type = []
        
        for cur_dict in self.opponent_bucket : 
            tmp_dict = dict()
            for v in cur_dict : 
                tmp_dict[v] = 2
            node_type.append(tmp_dict)
        
        for i in range(len(self.opponent_bucket)) : 
            v_pair1 = [0] * int(len(self.entire_pairs[i][0]) - 2*self.n_train[i]) # N - train - valid
            v_pair2 = copy.deepcopy(v_pair1)
            v_pair3 = copy.deepcopy(v_pair2)
            v_pair4 = copy.deepcopy(v_pair3)
            self.test_pos_samples.append((v_pair1, v_pair2))
            self.test_neg_samples.append((v_pair3, v_pair4))
        
        for i in range(len(self.opponent_bucket)) : # Sample Train / Valid
            cur_entire_pairs = set(range(len(self.entire_pairs[i][0])))
            train_pairs = list(np.random.choice(a = list(cur_entire_pairs), size = self.n_train[i], replace = False))
            
            for i1, idx in enumerate(train_pairs) : 
                v1 = self.entire_pairs[i][0][idx]
                v2 = self.entire_pairs[i][1][idx]
                self.train_pos_samples[i][0][i1] = v1
                self.train_pos_samples[i][1][i1] = v2
                node_type[i][v1] = 0 ; node_type[i][v2] = 0
            
            cur_entire_pairs = cur_entire_pairs - set(train_pairs)
            valid_pairs = list(np.random.choice(a = list(cur_entire_pairs), size = self.n_train[i], replace = False))
            
            for i1, idx in enumerate(valid_pairs) : 
                v1 = self.entire_pairs[i][0][idx]
                v2 = self.entire_pairs[i][1][idx]
                self.valid_pos_samples[i][0][i1] = v1
                self.valid_pos_samples[i][1][i1] = v2
                node_type[i][v1] = 1 ; node_type[i][v2] = 1
            
            test_pairs = cur_entire_pairs - set(valid_pairs)
            
            for i1, idx in enumerate(test_pairs) : 
                v1 = self.entire_pairs[i][0][idx]
                v2 = self.entire_pairs[i][1][idx]
                self.test_pos_samples[i][0][i1] = v1
                self.test_pos_samples[i][1][i1] = v2
            
        for i in range(len(self.opponent_bucket)) : # Create Negative Samples
            
            # Train Negative: Train-Train or Train-NonTrain
            neg_candid1 = self.train_pos_samples[i][0] + self.train_pos_samples[i][1]
            neg_candid2 = np.random.choice(a = list(node_type[i].keys()), size = len(neg_candid1), replace = False)
            candid_idx = 0 ; true_idx = 0
            max_idx = len(self.train_neg_samples[i][0])
            np.random.shuffle(neg_candid1)
            
            while (true_idx < max_idx) :
                
                vidx1 = neg_candid1[candid_idx]
                vidx2 = neg_candid2[candid_idx]
                
                if ((vidx1 != vidx2) & (self.opponent_bucket[i][vidx1] != vidx2)) : 
                    self.train_neg_samples[i][0][true_idx] = vidx1
                    self.train_neg_samples[i][1][true_idx] = vidx2
                    true_idx += 1
                
                if candid_idx == len(neg_candid1) - 1 : 
                    candid_idx = 0
                    np.random.shuffle(neg_candid1)
                    neg_candid2 = np.random.choice(a = list(node_type[i].keys()), size = len(neg_candid1), replace = False)
                else : 
                    candid_idx += 1
                    
            # Valid Negative: Train-Train or Train-NonTrain
            neg_candid1 = self.valid_pos_samples[i][0] + self.valid_pos_samples[i][1]
            neg_candid2 = np.random.choice(a = list(node_type[i].keys()), size = len(neg_candid1), replace = False)
            candid_idx = 0 ; true_idx = 0
            max_idx = len(self.valid_neg_samples[i][0])
            np.random.shuffle(neg_candid1)
            
            while (true_idx < max_idx) :
                
                vidx1 = neg_candid1[candid_idx]
                vidx2 = neg_candid2[candid_idx]
                
                if ((vidx1 != vidx2) & (self.opponent_bucket[i][vidx1] != vidx2)) : 
                    self.valid_neg_samples[i][0][true_idx] = vidx1
                    self.valid_neg_samples[i][1][true_idx] = vidx2
                    true_idx += 1
                
                if candid_idx == len(neg_candid1) - 1  : 
                    candid_idx = 0
                    np.random.shuffle(neg_candid1)
                    neg_candid2 = np.random.choice(a = list(node_type[i].keys()), size = len(neg_candid1), replace = False)
                else : 
                    candid_idx += 1
                    
            # Test Negative: Train-Train or Train-NonTrain
            neg_candid1 = []
            for v in node_type[i] : 
                if node_type[i][v] == 2 : 
                    neg_candid1.append(v)
            neg_candid2 = copy.deepcopy(neg_candid1)
            
            candid_idx = 0 ; true_idx = 0
            max_idx = len(self.test_neg_samples[i][0])
            np.random.shuffle(neg_candid1)
            np.random.shuffle(neg_candid2)
            
            while (true_idx < max_idx) :
                
                vidx1 = neg_candid1[candid_idx]
                vidx2 = neg_candid2[candid_idx]
                
                if ((vidx1 != vidx2) & (self.opponent_bucket[i][vidx1] != vidx2)) : 
                    self.test_neg_samples[i][0][true_idx] = vidx1
                    self.test_neg_samples[i][1][true_idx] = vidx2
                    true_idx += 1
                
                if candid_idx == len(neg_candid1) - 1  : 
                    candid_idx = 0
                    np.random.shuffle(neg_candid1)
                    np.random.shuffle(neg_candid2)
                else : 
                    candid_idx += 1

    def load_dataset(self, data_type) :
        
        ind1 = []
        ind2 = []
        labels = []
        if data_type == 'train':
            for i in range(len(self.train_pos_samples)) :
                ind1.append(self.train_pos_samples[i][0] + self.train_neg_samples[i][0])
                ind2.append(self.train_pos_samples[i][1] + self.train_neg_samples[i][1])
                size1, size2 = len(self.train_pos_samples[i][0]), len(self.train_pos_samples[i][1])
                label = torch.tensor([1.0] * size1 + [0.0] * size2).to(self.device)
                labels.append(label)
        elif data_type == 'valid':
            for i in range(len(self.valid_pos_samples)) :
                ind1.append(self.valid_pos_samples[i][0] + self.valid_neg_samples[i][0])
                ind2.append(self.valid_pos_samples[i][1] + self.valid_neg_samples[i][1])
                size1, size2 = len(self.valid_pos_samples[i][0]), len(self.valid_pos_samples[i][1])
                label = torch.tensor([1.0] * size1 + [0.0] * size2).to(self.device)
                labels.append(label)
        else:  # Test
            for i in range(len(self.test_pos_samples)) :
                ind1.append(self.test_pos_samples[i][0] + self.test_neg_samples[i][0])
                ind2.append(self.test_pos_samples[i][1] + self.test_neg_samples[i][1])
                size1, size2 = len(self.test_pos_samples[i][0]), len(self.test_pos_samples[i][1])
                label = torch.tensor([1.0] * size1 + [0.0] * size2).to(self.device)
                labels.append(label)

        return ind1, ind2, labels
    
## Task 2
    
class Task2DataLoader():

    def __init__(self, total_label, init_seed, device, sampling=True, split_ratio=10,
                 n_train_per_each=10000,
                 n_valid_per_each=10000,
                 n_test_per_each=10000):
        np.random.seed(init_seed)  ## Fixing seed
        torch.manual_seed(init_seed)  ## Fixing seed
        
        self.train_positive_pair = []
        self.train_negative_pair = []
        self.valid_positive_pair = []
        self.valid_negative_pair = []
        self.test_positive_pair = []
        self.test_negative_pair = []
        
        for label in tqdm(total_label) :
            
            labels = torch.tensor(label)
            self.N_NODE = labels.shape[0]
            self.device = device
            self.LABEL, self.LABEL_to_NODE = np.unique(labels.to('cpu').numpy(),
                                                       return_inverse=True)
            self.N_LABEL = self.LABEL.shape[0]

            self.truncated_label = np.setdiff1d(self.LABEL, np.array([-1]))

            self.train_venue = list(np.random.choice(a=self.truncated_label,
                                                     size=max(int(self.N_LABEL * (split_ratio)), 2),
                                                     replace=False))

            valid_candidate = np.setdiff1d(self.truncated_label, self.train_venue)
            self.valid_venue = list(np.random.choice(a=valid_candidate,
                                                     size=max(int(self.N_LABEL * (split_ratio)), 2),
                                                     replace=False))

            self.test_venue = list(np.setdiff1d(valid_candidate, self.valid_venue))

            self.train_venue.sort()
            self.valid_venue.sort()
            self.test_venue.sort()

            self.train_labelwise_taxonomy = {i: [] for i in self.train_venue}
            self.valid_labelwise_taxonomy = {i: [] for i in self.valid_venue}
            self.test_labelwise_taxonomy = {i: [] for i in self.test_venue}

            ## Index 0: Train / Index 1: Validation / Index 2: Test
            self.TrainValidTest_Indexer = {v : 0 for v in self.LABEL}

            for v in (self.LABEL):

                if np.isin(v, self.train_venue):
                    continue
                elif v == -1 : 
                    self.TrainValidTest_Indexer[v] = -1
                elif np.isin(v, self.valid_venue):
                    self.TrainValidTest_Indexer[v] = 1
                else:
                    self.TrainValidTest_Indexer[v] = 2

            for i in range(labels.shape[0]):

                cur_v_label = self.LABEL[self.LABEL_to_NODE[i]]

                if self.TrainValidTest_Indexer[cur_v_label] == -1 : 
                    continue
                elif self.TrainValidTest_Indexer[cur_v_label] == 0:
                    self.train_labelwise_taxonomy[cur_v_label].append(i)
                elif self.TrainValidTest_Indexer[cur_v_label] == 1:
                    self.valid_labelwise_taxonomy[cur_v_label].append(i)
                else:
                    self.test_labelwise_taxonomy[cur_v_label].append(i)

            train_positive_pair1, train_positive_pair2 = [], []
            valid_positive_pair1, valid_positive_pair2 = [], []
            test_positive_pair1, test_positive_pair2 = [], []

            self.entire_train_nodes = []
            self.entire_valid_nodes = []
            self.entire_test_nodes = []

            np.random.seed(init_seed)
            for i in (self.train_venue):
                cur_interest = self.train_labelwise_taxonomy[i]
                self.entire_train_nodes.extend(cur_interest)
                if (len(cur_interest)**2)/2 < n_train_per_each : 
                    result = list(combinations(cur_interest, 2))
                    for v1, v2 in result : 
                        train_positive_pair1.extend([v1])
                        train_positive_pair2.extend([v2])
                else : 
                    IDX1 = list(np.random.choice(a=cur_interest, size=n_train_per_each))
                    IDX2 = list(np.random.choice(a=cur_interest, size=n_train_per_each))

                    IDX_candid1 = list(np.random.choice(a=cur_interest, size=100*len(IDX1)))
                    IDX_candid2 = list(np.random.choice(a=cur_interest, size=100*len(IDX2)))

                    candid_idx = 0

                    for i in range(len(IDX1)):
                        while True:
                            if IDX1[i] != IDX2[i]:
                                break
                            else:
                                IDX1[i] = IDX_candid1[candid_idx]
                                IDX2[i] = IDX_candid2[candid_idx]
                                candid_idx += 1
                    train_positive_pair1.extend(IDX1)
                    train_positive_pair2.extend(IDX2)

            for i in (self.valid_venue):
                cur_interest = self.valid_labelwise_taxonomy[i]
                self.entire_valid_nodes.extend(cur_interest)

                if (len(cur_interest)**2)/2 < n_valid_per_each : 
                    result = list(combinations(cur_interest, 2))
                    for v1, v2 in result : 
                        valid_positive_pair1.extend([v1])
                        valid_positive_pair2.extend([v2])
                else : 
                    IDX1 = list(np.random.choice(a=cur_interest, size=n_valid_per_each))
                    IDX2 = list(np.random.choice(a=cur_interest, size=n_valid_per_each))

                    IDX_candid1 = list(np.random.choice(a=cur_interest, size=100*len(IDX1)))
                    IDX_candid2 = list(np.random.choice(a=cur_interest, size=100*len(IDX2)))

                    candid_idx = 0

                    for i in range(len(IDX1)):
                        while True:
                            if IDX1[i] != IDX2[i]:
                                break
                            else:
                                IDX1[i] = IDX_candid1[candid_idx]
                                IDX2[i] = IDX_candid2[candid_idx]
                                candid_idx += 1
                    valid_positive_pair1.extend(IDX1)
                    valid_positive_pair2.extend(IDX2)

            train_negative_pair1, train_negative_pair2 = train_positive_pair1[:], train_positive_pair2[:]
            valid_negative_pair1, valid_negative_pair2 = valid_positive_pair1[:], valid_positive_pair2[:]

            train_negative_pair2 = np.random.permutation(train_negative_pair2)
            valid_negative_pair2 = np.random.permutation(valid_negative_pair2)
            train_neg_candid = np.random.choice(self.entire_train_nodes, int(100 * len(train_positive_pair1)))
            valid_neg_candid = np.random.choice(self.entire_valid_nodes, int(100 * len(valid_positive_pair1)))
            

            neg_idx = -1
            pair_idx = -1

            for v1, v2 in (zip(train_negative_pair1, train_negative_pair2)):
                pair_idx += 1
                if self.LABEL_to_NODE[v1] == self.LABEL_to_NODE[v2]:
                    while True:
                        if neg_idx > train_neg_candid.shape[0] - 2 : 
                            train_neg_candid = np.random.choice(self.entire_train_nodes, int(100 * len(train_positive_pair1)))
                            neg_idx = 0
                        neg_idx += 1
                        if self.LABEL_to_NODE[v1] != self.LABEL_to_NODE[train_neg_candid[neg_idx]]:
                            train_negative_pair2[pair_idx] = train_neg_candid[neg_idx]
                            break
            

            neg_idx = -1
            pair_idx = -1
            for v1, v2 in (zip(valid_negative_pair1, valid_negative_pair2)):
                pair_idx += 1
                if self.LABEL_to_NODE[v1] == self.LABEL_to_NODE[v2]:
                    while True:
                        if neg_idx > valid_neg_candid.shape[0] - 2 : 
                            valid_neg_candid = np.random.choice(self.entire_valid_nodes, int(100 * len(train_positive_pair1)))
                            neg_idx = 0
                        neg_idx += 1
                        if self.LABEL_to_NODE[v1] != self.LABEL_to_NODE[valid_neg_candid[neg_idx]]:
                            valid_negative_pair2[pair_idx] = valid_neg_candid[neg_idx]
                            break
                        
            for i in (self.test_venue):
                cur_interest = self.test_labelwise_taxonomy[i]
                self.entire_test_nodes.extend(cur_interest)

                if (len(cur_interest)**2)/2 < n_test_per_each : 
                    result = list(combinations(cur_interest, 2))
                    for v1, v2 in result : 
                        test_positive_pair1.extend([v1])
                        test_positive_pair2.extend([v2])

                else : 

                    IDX1 = list(np.random.choice(a=cur_interest, size=n_test_per_each))
                    IDX2 = list(np.random.choice(a=cur_interest, size=n_test_per_each))

                    IDX_candid1 = list(np.random.choice(a=cur_interest, size=len(IDX1)))
                    IDX_candid2 = list(np.random.choice(a=cur_interest, size=len(IDX2)))

                    candid_idx = 0

                    for i in range(len(IDX1)):
                        while True:
                            if IDX1[i] != IDX2[i]:
                                break
                            else:
                                IDX1[i] = IDX_candid1[candid_idx]
                                IDX2[i] = IDX_candid2[candid_idx]
                                candid_idx += 1
                    test_positive_pair1.extend(IDX1)
                    test_positive_pair2.extend(IDX2)

            test_negative_pair1, test_negative_pair2 = test_positive_pair1[:], test_positive_pair2[:]
            test_negative_pair2 = np.random.permutation(test_negative_pair2)
            test_neg_candid = np.random.choice(self.entire_test_nodes, int(len(test_positive_pair1) * 10))

            neg_idx = -1
            pair_idx = -1
            for v1, v2 in (zip(test_negative_pair1, test_negative_pair2)):
                pair_idx += 1
                if self.LABEL_to_NODE[v1] == self.LABEL_to_NODE[v2]:
                    while True:
                        if neg_idx > test_neg_candid.shape[0] - 2 : 
                            test_neg_candid = np.random.choice(self.entire_test_nodes, int(len(test_positive_pair1) * 10))
                            neg_idx = 0
                        neg_idx += 1
                        if self.LABEL_to_NODE[v1] != self.LABEL_to_NODE[test_neg_candid[neg_idx]]:
                            test_negative_pair2[pair_idx] = test_neg_candid[neg_idx]
                            break

            self.train_positive_pair.append(np.array([train_positive_pair1, train_positive_pair2]))
            self.train_negative_pair.append(np.array([train_negative_pair1, train_negative_pair2]))
            self.valid_positive_pair.append(np.array([valid_positive_pair1, valid_positive_pair2]))
            self.valid_negative_pair.append(np.array([valid_negative_pair1, valid_negative_pair2]))
            self.test_positive_pair.append(np.array([test_positive_pair1, test_positive_pair2]))
            self.test_negative_pair.append(np.array([test_negative_pair1, test_negative_pair2]))
    
    def full_train_pair_loader(self) : 
        
        entire_ind1 = []
        entire_ind2 = []
        entire_label = []
        
        for interest_pos_set, interest_neg_set in zip(self.train_positive_pair, self.train_negative_pair) : 
            
            a1 = interest_pos_set[0].shape[0]
            a2 = interest_neg_set[0].shape[0]
            
            IND1 = list(interest_pos_set[0]) + list(interest_neg_set[0])
            IND2 = list(interest_pos_set[1]) + list(interest_neg_set[1])

            label = torch.tensor(([1.0] * a1) + ([0.0] * a2)).to(self.device)
            entire_ind1.append(IND1)
            entire_ind2.append(IND2)
            entire_label.append(label)
            
        return entire_ind1, entire_ind2, entire_label
    
    def batch_train_pair_loader(self) : 
        
        entire_ind1 = []
        entire_ind2 = []
        entire_label = []
        
        for interest_pos_set, interest_neg_set in zip(self.train_positive_pair, self.train_negative_pair) : 
            
            a1 = interest_pos_set[0].shape[0]
            a2 = interest_neg_set[0].shape[0]
            
            IND1 = list(interest_pos_set[0]) + list(interest_neg_set[0])
            IND2 = list(interest_pos_set[1]) + list(interest_neg_set[1])

            label = torch.tensor(([1.0] * a1) + ([0.0] * a2)).to(self.device)
            entire_ind1.append(IND1)
            entire_ind2.append(IND2)
            entire_label.append(label)
            
        return entire_ind1, entire_ind2, entire_label
    

    def partial_pair_loader(self, batch_size, seed, device='cpu', mode='train'):
        
        #batch_of_pair = []
        entire_ind1 = []
        entire_ind2 = []
        entire_label = []
        
        if mode == 'train':
            entire_pos_set = self.train_positive_pair
            entire_neg_set = self.train_negative_pair
        elif mode == 'valid':
            entire_pos_set = self.valid_positive_pair
            entire_neg_set = self.valid_negative_pair
        else:
            entire_pos_set = self.test_positive_pair
            entire_neg_set = self.test_negative_pair
            
        for interest_pos_set, interest_neg_set in zip(entire_pos_set, entire_neg_set) : 

            np.random.seed(seed)
            if int(batch_size / 2) < min(interest_pos_set.shape[1], interest_neg_set.shape[1]):

                pos_idxs = np.random.choice(a=np.arange(interest_pos_set.shape[1]),
                                            size=int(batch_size / 2),
                                            replace=False)
                neg_idxs = np.random.choice(a=np.arange(interest_neg_set.shape[1]),
                                            size=int(batch_size / 2),
                                            replace=False)

                IND1 = list(interest_pos_set[0, pos_idxs]) + list(interest_neg_set[0, neg_idxs])
                IND2 = list(interest_pos_set[1, pos_idxs]) + list(interest_neg_set[1, neg_idxs])
                N_pos, N_neg = int(batch_size / 2), int(batch_size / 2)

            else:
                IND1 = list(interest_pos_set[0]) + list(interest_neg_set[0])
                IND2 = list(interest_pos_set[1]) + list(interest_neg_set[1])
                N_pos, N_neg = int(interest_pos_set[0].shape[0]), int(interest_neg_set[0].shape[0])

            #batch_of_pair.append([IND1, IND2, label])
            entire_ind1.append(IND1)
            entire_ind2.append(IND2)
            
            label = torch.tensor([1.0] * int(N_pos) + [0.0] * int(N_neg)).to(device)
            entire_label.append(label)

        return entire_ind1, entire_ind2, entire_label
    
def create_batch_pairs(positive_pairs = None, negative_pairs = None, 
                       valid_Vs1 = None, valid_Vs2 = None, valid_labels = None, target_time = 0,
                       batch_size = 50000, device = 'cuda:0', data_type = 'train', seed = 0) : 
    
    np.random.seed(seed)
    
    if data_type == 'train' : 
        Indexer = []
        Indicator1 = []
        Indicator2 = []
        
        for t in (range(len(positive_pairs))) : 
            
            curP1 = positive_pairs[t][0]
            curP2 = positive_pairs[t][1]
            curN1 = negative_pairs[t][0]
            curN2 = negative_pairs[t][1]
                
            if len(curP1) < batch_size : # Directly doing this!
                
                Indicator1.append(list(curP1) + list(curN1))
                Indicator2.append(list(curP2) + list(curN2))
                Indexer.append([]) # Does not need re
                
            else : 
                order_indexer = np.arange(len(curP1))
                np.random.shuffle(order_indexer)
                
                NB = int(len(curP2) // batch_size) + 1
                part_batch1 = []
                part_batch2 = []
                part_indicator = []
                
                for b in range(NB) : 
                    ID = order_indexer[int(b * batch_size) : int((b+1) * batch_size)]
                    cur_batch_p1 = curP1[ID]
                    cur_batch_p2 = curP2[ID]
                    cur_batch_n1 = curN1[ID]
                    cur_batch_n2 = curN2[ID]
                    curV = set(list(cur_batch_p1) + list(cur_batch_p2) + list(cur_batch_n1) + list(cur_batch_n2))
                    V2ID = {v : i for i, v in enumerate(curV)}
                    b1 = [V2ID[v] for v in cur_batch_p1] + [V2ID[v] for v in cur_batch_n1]
                    b2 = [V2ID[v] for v in cur_batch_p2] + [V2ID[v] for v in cur_batch_n2]
                    orderedV = list(V2ID.keys())
                    part_batch1.append(b1)
                    part_batch2.append(b2)
                    part_indicator.append(orderedV)
            
                Indicator1.append(part_batch1)
                Indicator2.append(part_batch2)
                Indexer.append(part_indicator)
                
        return Indicator1, Indicator2, Indexer
                    
    elif (data_type == 'valid') or (data_type == 'test') : 
        
        Indicator1 = []
        Indicator2 = []
        Indexer = []
        label = []
        
        #valid_Vs1, valid_Vs2
        #print(valid_Vs1)
        curV1 = valid_Vs1[target_time]
        curV2 = valid_Vs2[target_time]
        curY = list(valid_labels[target_time].to('cpu'))
        
        for b in range(int(len(curV1)//batch_size) + 1) : 
            
            tmpV1 = curV1[int(b * batch_size) : int((b + 1) * batch_size)]
            tmpV2 = curV2[int(b * batch_size) : int((b + 1) * batch_size)]
            tmpVs = {v : i for i, v in enumerate(set(tmpV1 + tmpV2))}
            
            Indicator1.append([tmpVs[v] for v in tmpV1])
            Indicator2.append([tmpVs[v] for v in tmpV2])
            Indexer.append(list(tmpVs.keys()))
        
        return Indicator1, Indicator2, Indexer, curY
        
    else : 
        raise TypeError("Data type should be given either train/valid/test.")
        
        
    
def give_window_stride(final, w_size) : 
    
    each_buckets = []
    
    for i in range(final) : 
        if i < w_size - 1 : 
            each_buckets.append([t for t in range(i + 1)])
        else : 
            each_buckets.append([t + 1 for t in range(i - w_size, i)])
        
    return each_buckets

def modify_in_and_out(Xs, EIDXs, target_batch, device) : 
    
    is_cpu = Xs[0].get_device() # Device type = -1 : CPU
    grad_of_mat = Xs[0].requires_grad # Do we need gradient for the following matrices?
    uniqueV = []
    for e in target_batch : 
        uniqueV.extend(EIDXs[e])
    
    uniqueV = set(uniqueV) # Unique Nodes of Target Indices...
    reindexer = {v : i for i, v in enumerate(uniqueV)} # New indices of Nodes...
            
    orig2new = []
    new2orig = []
    for e in target_batch :     
        cur_index = [reindexer[v] for v in EIDXs[e]]
        orig2new.append(cur_index)
    
    newX = []
    for i, e in enumerate(target_batch) :     
        cur_index = orig2new[i]
        if is_cpu : 
            curX = torch.zeros((len(reindexer), Xs[e].shape[1]))
            curX[cur_index, :] = Xs[e]
            curX = curX.to(device)
        else : 
            curX = torch.zeros((len(reindexer), Xs[e].shape[1])).to(device)
            curX[cur_index, :] = Xs[e]
        curX.requires_grad = grad_of_mat
        newX.append(curX) # We stack re-indexed feature matrix
    
    return newX, orig2new

def simply_evaluate(pred, answer) : 
    
    pred = pred.to('cpu').detach().numpy()
    answer = answer.to('cpu').numpy()
    
    return average_precision_score(answer, pred), roc_auc_score(answer, pred)

def train_LSTM_task1(model, Xs, loader, train_triple, valid_triple, test_triple, lr = 0.001, epochs = 100, device = 'cuda:0', target_time = 9) : 
    
    model.train() # classifier
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 1e-6)
    criterion = torch.nn.BCELoss()
    _, _, train_label = loader.load_dataset("train")
    _, _, valid_label = loader.load_dataset("valid")
    _, _, test_label = loader.load_dataset("test")
    
    valid_label = valid_label[target_time]
    test_label = test_label[target_time]
    
    valid_ind1, valid_ind2, valid_v = valid_triple[0][target_time], valid_triple[1][target_time], valid_triple[2][target_time]
    test_ind1, test_ind2, test_v = test_triple[0][target_time], test_triple[1][target_time], test_triple[2][target_time]
    valid_acc = 0
    for ep in tqdm(range(epochs)) : 
        
        model.train()
        
        for t in range(target_time + 1) : 
            
            ind1, ind2, reindex = train_triple[0][t], train_triple[1][t], train_triple[2][t]
            curY = train_label[t].to(device)
            curX = Xs[t][:, reindex, :].to(device)
            optimizer.zero_grad()
            pred = model(curX, ind1, ind2)
            loss = criterion(pred, curY)
            loss.backward()
            optimizer.step()
            
            del curX, curY
            
        if (ep + 1) % 5 == 0 :  # batch is a last batch = Evaluation time
            cur_acc = task_mlp_evaluator(model, Xs[target_time][:, valid_v, :].to(device), valid_ind1, valid_ind2, valid_label)[1]
            if valid_acc < cur_acc :
                param = copy.deepcopy(model.state_dict())
                valid_acc = cur_acc

    model.load_state_dict(param)
    testX = Xs[target_time][:, test_v, :].to(device)
    test_acc = task_mlp_evaluator(model, testX, test_ind1, test_ind2, test_label)[1]
        
    return test_acc
            

def train_LSTM_task2(model, Xs, loader, evaluation_time, lr = 0.001, epochs= 300, device = 'cuda:0', 
                   n_batch = 50000, n_valid = 100000, n_test = 1000000, seed = 0, batch_size = None, 
                    I1 = None, I2 = None, Vs = None, V1 = None, V2 = None, VV = None, VY = None, 
                                                    T1 = None, T2 = None, TV = None, TY = None) : 
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 1e-6)
    criterion = torch.nn.BCELoss()
    
    if batch_size is None : # No batch
        
        train_indices1, train_indices2, train_labels = loader.full_train_pair_loader() # Loading Full Datasets
        valid_indices1, valid_indices2, valid_labels = loader.partial_pair_loader(batch_size = n_valid, seed = seed, device='cpu', mode='valid')
        test_indices1, test_indices2, test_labels = loader.partial_pair_loader(batch_size = n_test, seed = seed, device='cpu', mode='test')

        valid_acc = 0
        for ep in tqdm(range(epochs)) : 

            model.train()

            for batch in range(evaluation_time + 1) :

                curX = Xs[batch].to(device)

                ind1, ind2 = train_indices1[batch], train_indices1[batch]
                curY = train_labels[batch]

                optimizer.zero_grad()
                pred = model(curX, ind1, ind2)
                loss = criterion(pred, curY)
                loss.backward()
                optimizer.step()

                del curX

            if (ep + 1) % 5 == 0 :  # batch is a last batch = Evaluation time
                ind1, ind2, label = valid_indices1[batch], valid_indices2[batch], valid_labels[batch]
                cur_acc = task_mlp_evaluator(model, Xs[batch].to(device), ind1, ind2, label)[1]
                if valid_acc < cur_acc :
                    param = copy.deepcopy(model.state_dict())
                    valid_acc = cur_acc

        model.load_state_dict(param)
        testX = Xs[batch].to(device)
        ind1, ind2, label = test_indices1[batch], test_indices2[batch], test_labels[batch]
        test_acc = task_mlp_evaluator(model, testX, ind1, ind2, label)
        
    else : # Do batch
       
        valid_acc = 0
        curY = torch.zeros(1).to(device)
        for ep in tqdm(range(epochs)) : 
            
            model.train()
            
            if len(I1) > 0 : 
                
                rand1 = int(np.random.choice(a = np.arange(len(I1)), size = 1))
                i1, i2, vv = I1[rand1], I2[rand1], Vs[rand1]
            
            else : 
                i1, i2, vv = I1, I2, Vs
                
            for t in range(evaluation_time + 1) :

                subi1, subi2, subv = i1[t], i2[t], vv[t]
                
                if len(subv) == 0 : # No batch in this case
                    optimizer.zero_grad()
                    curZ = Xs[t].to(device)
                    pred = model(curZ, subi1, subi2)
                    curY = torch.tensor([1.0] * int(pred.shape[0]/2) + [0.0] * int(pred.shape[0] - pred.shape[0]/2)).to(device)
                    loss = criterion(pred, curY)
                    loss.backward()
                    optimizer.step()
                    del curZ
                
                else : # Do batch training
                    for t2 in range(len(subi1)) : 
                        si1, si2, sv = subi1[t2], subi2[t2], subv[t2]
                        optimizer.zero_grad()
                        curZ = Xs[t][:, sv, :].to(device)
                        pred = model(curZ, si1, si2)
                        if curY.shape[0] != pred.shape[0] : 
                            curY = torch.tensor([1.0] * int(pred.shape[0]/2) + [0.0] * int(pred.shape[0] - pred.shape[0]/2)).to(device)
                        else : 
                            continue
                        loss = criterion(pred, curY)
                        loss.backward()
                        optimizer.step()
                        del curZ
            
            if (ep + 1) % 5 == 0 : 
                cur_acc = make_batch_predictions(model, Xs[evaluation_time], V1, V2, VV, VY, device = device)
                if valid_acc < cur_acc :
                    param = copy.deepcopy(model.state_dict())
                    valid_acc = cur_acc
        
        model.load_state_dict(param)
        testX = Xs[evaluation_time]
        test_acc = make_batch_predictions(model, testX, T1, T2, TV, TY, device = device)
                    
    return test_acc

def make_batch_predictions(model, Z, B1, B2, Vs, labels, device) : 
    
    predictions = []
    
    with torch.no_grad() : 
        
        model.eval()
    
        for t in range(len(B1)) : 

            if len(B1[t]) == 0 : 
                continue
            
            else : 
                ind1 = B1[t]
                ind2 = B2[t]
                vidx = Vs[t]
                curZ = Z[:, vidx, :].to(device)
                pred = model(curZ, ind1, ind2)
                pred = list(pred.to('cpu').numpy())
                predictions.extend(pred)
                del curZ
    
    return roc_auc_score(labels, predictions)

def reindex_indices(I1, I2) : 
    newI1 = []
    newI2 = []
    newVs = []
    
    for i1, i2 in zip(I1, I2) : 
        
        curV = set(i1 + i2)
        Vdict = {v : i for i, v in enumerate(curV)}
        newi1 = [Vdict[v] for v in i1]
        newi2 = [Vdict[v] for v in i2]
        newV = list(Vdict.keys())
        newI1.append(newi1)
        newI2.append(newi2)
        newVs.append(newV)
        
    return newI1, newI2, newVs

def task_mlp_evaluator(model, X, ind1, ind2, label) :
    
    with torch.no_grad() : 
        model.eval()
        pred = model(X, ind1, ind2).to('cpu').detach().numpy()
        if isinstance(label, list) : 
            label = np.array(label)
        else: 
            label = label.to('cpu').numpy()
        
        return average_precision_score(label, pred), roc_auc_score(label, pred)
