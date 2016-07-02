import csv
import numpy as np

TRAIN_PATH = 'numerai_training_data.csv'
PREDICT_PATH = 'numerai_tournament_data.csv'
RESULTS = 'numerai_results.csv'


def get_input():
    train_data = []
    train_target = []
    with open(TRAIN_PATH) as csvfile:
        spamreader = csv.reader(csvfile)
        i=0
        for row in spamreader:
            if i == 0:
                i+=1
                continue
            d = map(float,row[:-1])
            d = [[i] for i in d]
            if int(row[-1])==1:
                t = [0,1]
            else:
                t = [1,0]
            train_data.append(d)
            train_target.append(t)
    #Dimensions for train_data is batch_size * 21 * 1
    #Dimensions for train target is batch_size * 2
    return train_data,train_target


def get_logi_reg():
    train_data = []
    train_target = []
    with open(TRAIN_PATH) as csvfile:
        spamreader = csv.reader(csvfile)
        i=0
        for row in spamreader:
            if i == 0:
                i+=1
                continue
            d = map(float,row[:-1])
            if int(row[-1])==1:
                t = [0,1]
            else:
                t = [1,0]
            train_data.append(d)
            train_target.append(t)
    #Dimensions for train_data is batch_size * 21
    #Dimensions for train target is batch_size * 2
    return train_data,train_target

def get_pred_log():
    ids = []
    train_data = []
    with open(PREDICT_PATH) as csvfile:
        spamreader = csv.reader(csvfile)
        i=0
        for row in spamreader:
            if i == 0:
                i+=1
                continue
            id = row[0]
            d = map(float,row[1:])
            ids.append(id)
            train_data.append(d)
    return ids,train_data


def get_pred():
    ids = []
    train_data = []
    with open(PREDICT_PATH) as csvfile:
        spamreader = csv.reader(csvfile)
        i=0
        for row in spamreader:
            if i == 0:
                i+=1
                continue
            id = row[0]
            d = map(float,row[1:])
            d = [[i] for i in d]
            ids.append(id)
            train_data.append(d)
    return ids,train_data


def save_output(id,pred,model):
    with open('results/'+str(model)+'-'+RESULTS, 'w') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(['t_id','probability'])
        # print id[0],pred[0]
        for i,v in enumerate(id):
            spamwriter.writerow([str(v),str(pred[i][1])])


def get_input_hybrid():
    train_data = []
    train_target = []
    with open(TRAIN_PATH) as csvfile:
        spamreader = csv.reader(csvfile)
        i=0
        for row in spamreader:
            if i == 0:
                i+=1
                continue
            d = map(float,row[:-1])
            d = [[i] for i in d]
            if int(row[-1])==1:
                t = [1]
            else:
                t = [0]
            train_data.append(d)
            train_target.append(t)
    #Dimensions for train_data is batch_size * 21 * 1
    #Dimensions for train target is batch_size * 2
    return train_data,train_target

def save_output_hybrid(id,pred,model):
    with open('results/'+str(model)+'-'+RESULTS, 'w') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(['t_id','probability'])
        # print id[0],pred[0]
        for i,v in enumerate(id):
            spamwriter.writerow([str(v),str(pred[i][0])])