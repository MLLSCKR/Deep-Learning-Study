"""
Date 220616.
    MLP 직접 구현해 보기 with using class. (by SCL)
        VER01(220616. only for regression)
"""

import os
import csv
import numpy as np

# dataset class 선언
class Regression_Dataset(object):
    
    def __init__(self, name, mode):
        self.name = name
        self.mode = mode
        
        file_path = "C:/Users/user/Desktop/DeepLearning/MLP_CLASS/abalone.csv"
        
        with open(file_path) as csvfile:
            csvreader = csv.reader(csvfile)
            headers = None
            
            headers = next(csvreader, None)
            rows = []
            for row in csvreader:
                rows.append(row)
        
        self.xs = np.zeros([len(rows), 10])
        self.ys = np.zeros([len(rows), 1])
        
        for n, row in enumerate(rows):
            if row[0] == 'I':
                self.xs[n, 0] = 1
            elif row[0] == 'M':
                self.xs[n, 1] = 1
            elif row[0] == 'F':
                self.xs[n, 2] = 1
            
            self.xs[n, 3:] = row[1:-1]
            self.ys[n, :] = row[-1]
    
    def train_count(self):
        return len(self.tr_X)
    
    def shuffle_dataset(self, tr, te, val):
        # randomly shuffle dataset(self.xs, self.ys) and define self.tr, self.te, self.val
        self.dataset_indices = np.arange(self.xs.shape[0])
        np.random.shuffle(self.dataset_indices)
        
        dataset_len = self.xs.shape[0]
        
        tr_cnt = int(dataset_len * tr)
        te_cnt = int(dataset_len * te)
        val_cnt = int(dataset_len * val)
        
        self.tr_X = self.xs[self.dataset_indices[:tr_cnt]]
        self.tr_Y = self.ys[self.dataset_indices[:tr_cnt]]
        self.te_X = self.xs[self.dataset_indices[tr_cnt + 1:tr_cnt + te_cnt]]
        self.te_Y = self.ys[self.dataset_indices[tr_cnt + 1:tr_cnt + te_cnt]]
        self.val_X = self.xs[self.dataset_indices[tr_cnt + te_cnt + 1:tr_cnt + te_cnt + val_cnt]]
        self.val_Y = self.ys[self.dataset_indices[tr_cnt + te_cnt + 1:tr_cnt + te_cnt + val_cnt]]
    
    def shuffle_train_dataset(self, size):
        self.indices = np.arange(size)
        np.random.shuffle(self.indices)
        
    def get_train_data(self, batch_size, iteration_num):
        # batch size, iteration_num을 이용하여 현재 학습에 이용할 batch group data를 return한다.
        return self.tr_X[batch_size * iteration_num + 1: batch_size * (iteration_num + 1)], \
            self.tr_Y[batch_size * iteration_num + 1: batch_size * (iteration_num + 1)]
            
    def get_validate_data(self):
        self.val_indices = np.arange(len(self.val_X))
        np.random.shuffle(self.val_indices)
        
        va_X = self.val_X
        va_Y = self.val_Y
        
        return va_X, va_Y
    
    def get_test_data(self):
        te_X = self.te_X
        te_Y = self.te_Y
        
        return te_X, te_Y
    
    def train_prt_result(self, current_epoch, costs, accs, acc, tm1, tm2):
        print('Epoch {} : cost = {:5.3f}, accuracy = {:5.3f}/{:5.3f} {}/{} secs'.format(current_epoch, np.mean(costs), np.mean(accs), acc, tm1, tm2))
    
    def test_prt_result(self, name, acc, tm1):
        print('Model {} test report : accuracy = {:5.3f}, {}secs'.format(name, acc, tm1))
    
    def forward_postproc(self, output, tr_Y):
        # regression dataset
        # LOSS : MSE
        #   output ; mx1 vector, tr_Y ; mx1 vector
        loss = np.mean(np.square(output - tr_Y))
        aux = output - tr_Y
        
        # aux will be used for backpropagation
        return loss, aux
    
    def backprop_postproc(self, G_loss, aux):
        diff = aux
        shape = diff.shape
        
        g_loss_square = np.ones(shape) / np.prod(shape)
        g_square_diff = 2*diff
        g_diff_output = 1
        
        G_square = g_loss_square * G_loss
        G_diff = g_square_diff * G_square
        G_output = g_diff_output * G_diff
        
        return G_output
        
    def eval_accuracy(self, x, y, output):
        mse = np.mean(np.square(output - y))
        accuracy = 1 - np.sqrt(mse) / np.mean(y)
        
        return accuracy
    
    def get_visualize_data(self, cnt):
        idx = np.random.choice(self.indices, cnt)
        X = self.xs[idx]
        Y = self.ys[idx]
        
        return X, Y
    
    def visualize(self, x, y, est):
        acc = self.eval_accuracy(x, y, est)
        
        for i in range(len(x)):
            print('input : {}, estimate result : {}, real value : {}'.format(x[i], est[i], y[i]))
        print('accuracy : {}'.format(acc))