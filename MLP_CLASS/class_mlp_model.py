"""
Date 220616.
    MLP 직접 구현해 보기 with using class. (by SCL)
        VER01(220616. only for regression)
"""

import os
import time
import numpy as np
from mathutil import *

# mlp class 선언
class mlp_model(object):
    
    def __init__(self, name, hconfigs, learning_type, dataset):
        self.name = name
        self.learning_type = learning_type
        
        self.dataset = dataset
        # dataset은 달리 선언한 class이다.
        # 위 class(dataset)은 tr_X, tr_Y, te_X, te_Y, val_X, val_Y 라는 class attribute를 지닌다.
        
        # use layers_para_generate function to produce weight matrixs and bias vectors
        #   to call layers_para_generate function, input tr_X, tr_Y[traning data input, output], hconfigs[list type]
        # 
        #       result of layers_param_generate function(self.pm)
        #       self.pm is dictionary type
        #           {'key[layer #]' : [weight_matrix, bias_matrix], ...}
        self.layers_param_generate(hconfigs, self.dataset.xs.shape[-1],\
                                   self.dataset.ys.shape[-1])
    
    def __str__(self):
        print('model {} brief information. {}, training data : {}, test data :\
              {}, validation data : {}'.format(self.name, self.learning_type, \
            self.dataset.tr_X.shape[0], self.dataset.te_X.shape[0], \
            self.dataset.val_X.shape[0]))
    
    def __exec__(self, epoch_count = 10, batch_size = 10, learning_rate = 0.001, report = 0, cnt = 3):
        # for model training, use self.dataset.tr_X, self.dataset.tr_Y, self.pm
        """
        Deep learning 학습 절차
            training
                function call 순서
                    1. forward_neuralnet
                    2. forward_postproc
                    3. backprop_postproc
                    4. backprop_neuralnet
            학습에 필요한 각 함수의 세부적인 내용은 어떠한 학습인지에 따라 변하기에, class dataset(object)에서 별도로 정의하여 사용한다.
        """
        self.mlp_train(epoch_count, batch_size, learning_rate, report)
        
        # for model testing, use self.dataset.te_X, self.dataset.tr_Y, self.pm
        self.mlp_test()
        
        # for model visualization(show results), use self.dataset.te_X, 
        # self.dataset.te_Y, self.pm
        self.visualization(cnt)

    def layers_param_generate(self, hconfigs, input_shape, output_shape):
        self.hconfigs = hconfigs
        self.pm_hiddens = []
        
        for i in range(len(hconfigs) + 1):
            if i == 0:
                pre_cnt = input_shape
                aft_cnt = hconfigs[i]
            elif i == len(hconfigs):
                pre_cnt = hconfigs[i - 1]
                aft_cnt = output_shape
            else:
                pre_cnt = hconfigs[i - 1]
                aft_cnt = hconfigs[i]
            
            weight = np.random.normal(0, 0.030, [pre_cnt, aft_cnt])
            bias = np.zeros(aft_cnt)
            
            self.pm_hiddens.append({'w' : weight, 'b' : bias})


    def mlp_train(self, epoch_count = 10, batch_size = 10, \
                  learning_rate = 0.001, report = 0):
        self.learning_rate = learning_rate
        
        # random shuffle 후, trainin, test, validation set을 해당 비중으로 조정
        # self.tr_X, tr_Y, val_X, val_Y, te_X, te_Y에 저장됨
        self.dataset.shuffle_dataset(0.6, 0.2, 0.2)
        
        batch_count = int(self.dataset.tr_X.shape[0] / batch_size)
        
        print("model {} traiing is started".format(self.name))
        
        time_start, time_temp  = time.time(), time.time()
        
        for i in range(epoch_count):
            costs = []
            accs = []
            
            # only treat batch_size * batch_count number of data
            #   to avoid not divided error
            #
            #   dataset.shuffle_dataset function
            #       작동 원리 요약
            #       self.indices = np.arange(batch_size * batch_count)
            #       np.random.shuffle(self.indices)
            #           무작위로 섞인 index들의 정보가 self.dataset.indices에 저장되어 있음
            self.dataset.shuffle_train_dataset(batch_size * batch_count)
            
            # training
            for j in range(batch_count):
                trX, trY = self.dataset.get_train_data(batch_size, j)
                
                # forward propagation
                output, aux_nn = self.forward_neuralnet(trX)
                # output = X * W
                # aux_nn = X
                loss, aux_pp = self.forward_postproc(output, trY)
                # loss = cost function
                # aux_pp = 
                
                accuracy = self.eval_accuracy(trX, trY, output)
                
                # backward propagation
                G_loss=  1.0
                G_output = self.backprop_postproc(G_loss, aux_pp)
                self.backprop_neuralnet(G_output, aux_nn)
                
                costs.append(loss)
                accs.append(accuracy)
            
            # validation
            if report > 0 and (i + 1) % report == 0:
                vaX, vaY = self.dataset.get_validate_data()
                acc = self.eval_accuracy(vaX, vaY)
                time_mid = time.time()
                
                self.dataset.train_prt_result(i + 1, costs, accs, acc, time_mid - time_temp, time_mid - time_start)
                
                time_temp = time_mid
                
        time_end = time.time()
        print('Model {} train ended in {} secs'.format(self.name, time_end - time_start))
        
    def mlp_test(self):
       
        print("model {} test is started".format(self.name))
        
        start_time = time.time()
        
        teX, teY = self.dataset.get_test_data()
        
        # forward propagation
        output, aux_nn = self.forward_neuralnet(teX)
        accuracy = self.eval_accuracy(teX, teY, output)
        
        end_time = time.time()
        
        self.dataset.test_prt_result(self.name, accuracy, end_time - start_time)
        
    
    def eval_accuracy(self, x, y, output = None):
        if output is None:
            output, _ = self.forward_neuralnet(x)
            
        accuracy = self.dataset.eval_accuracy(x, y, output)
            
        return accuracy

    def forward_neuralnet(self, x):
        aux_nn = []
        
        temp_x = x
        for n, pm in enumerate(self.pm_hiddens):
            
            temp_y = np.matmul(temp_x, pm['w']) + pm['b']
            
            if n != (len(self.pm_hiddens) - 1):
                output = relu(temp_y)
                aux_nn.append([temp_x, output])
                temp_x = output
            else:
                output = temp_y
                aux_nn.append([temp_x, output])
        
        return output, aux_nn

    # by using output(calculated from forward_neuralnet), estimate result
    def forward_postproc(self, output, trY):
        loss, aux_pp = self.dataset.forward_postproc(output, trY)
        
        return loss, aux_pp
    
    # backprop postproc -> dL / dY, backprop postproc -> dL / dY * dY / dW
    def backprop_postproc(self, G_loss, aux):
        # aux : diff(output - y)
        G_output = self.dataset.backprop_postproc(G_loss, aux)
        
        return G_output
    
    
    def backprop_neuralnet(self, G_output, aux_nn):
        # aux_nn = [X_1st, X_2nd, ...]
        first = 1
        
        for n in reversed(range(len(self.pm_hiddens))):
            if first == 1:
                G_y = G_output
                x, y = aux_nn[n]
                
                first = 0
            else:
                x, y = aux_nn[n]
                G_y = derv_relu(y) * G_y
            
            g_y_weight = x.transpose()
            g_y_input = self.pm_hiddens[n]['w'].transpose()
            
            G_weight = np.matmul(g_y_weight, G_y)
            G_bias = np.sum(G_y, axis = 0)
            G_input = np.matmul(G_y, g_y_input)
            
            # updating weight
            self.pm_hiddens[n]['w'] -= self.learning_rate * G_weight
            
            # updating bias
            self.pm_hiddens[n]['b'] -= self.learning_rate * G_bias
            
            G_y = G_input
    
    def visualization(self, cnt):
        print('Model {} visualization'.format(self.name))
        
        deX, deY = self.dataset.get_visualize_data(cnt)
        est = self.get_estimate(deX)
        self.dataset.visualize(deX, deY, est)
        
    def get_estimate(self, X):
        output, _ = self.forward_neuralnet(X)
        
        return output

