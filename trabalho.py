import os
import numpy as np
import torch
import torch.utils.data as tchdata
from sklearn import preprocessing
from sklearn import metrics
from datetime import datetime
import matplotlib.pyplot as plt
class AccMectric(object):
    def __init__(self):
        self.reset()        
    def reset(self):
        self._sum = 0
        self._count = 0    
    def update(self, targets, outputs):
        pred = outputs.argmax(axis=1)
        self._sum += (pred == targets).sum()
        self._count += targets.shape[0]        
    def get(self):
        return self._sum / self._count
    
def plots(first, y_label, title, num_iterations , x_label='iterations',second=None, third=None, show_min=True):
    fig = plt.figure()
    fig.gca().set_position((.15, .3, .80, .6))    
    plt.ylabel(y_label+"->")    
    t =np.linspace(0, num_iterations-1, num=num_iterations)
    plt.figtext(.5,.92,title, fontsize=14, ha='center', fontweight='bold')        
    plt.figtext(0.25, .33, x_label+'->', fontsize=10,ha='left', va='center')  
    first_list, first_list_name=first
    first_array = np.array(first_list)
    (first_max_x, first_max_y)=(round((np.argmax(first_array)),3), round(np.amax(first_array), 3))
    if (second is not None):
        second_list, second_list_name=second
        second_array = np.array(second_list)
        (second_max_x, second_max_y)=(round((np.argmax(second_array)),3), round(np.amax(second_array), 3)) 
    if not(third is None):
        third_list, third_list_name=third
        third_array=np.array(third_list)
        (third_max_x, third_max_y)=(round((np.argmax(third_array)),3), round(np.amax(third_array), 3))
    if(show_min):
        plt.scatter(first_max_x,first_max_y,c='b',label='max_'+first_list_name+'('+str(first_max_x)+','+str(first_max_y)+')')
        if ((third is None) and (not(second is None))):
            plt.scatter(second_max_x,second_max_y,c='r',label='max_'+second_list_name+'('+str(second_max_x)+','+str(second_max_y)+')')
        elif (not(third is None) and not(second is None)):
            plt.scatter(second_max_x,second_max_y,c='r',label='max_'+second_list_name+'('+str(second_max_x)+','+str(second_max_y)+')')
            plt.scatter(third_max_x,third_max_y,c='g',label='max_'+third_list_name+'('+str(third_max_x)+','+str(third_max_y)+')')
    plot1,=plt.plot(t, np.squeeze(first_list), 'b-', linewidth=1.5, label=first_list_name)
    plt.legend(bbox_to_anchor=(0.4, -0.15))
    if (not(second is None)):
            plot2,=plt.plot(t, np.squeeze(second_list), 'r-', linewidth=1.5, label=second_list_name)
            plt.legend(bbox_to_anchor=(0.4, 0))
            if(not(third is None)):
                plot3,=plt.plot(t, np.squeeze(third_list), 'g-', linewidth=1.5, label=third_list_name)
                plt.legend(bbox_to_anchor=(0.4, 0.15))   
    
    pasta=os.path.dirname(__file__)     
    plt.savefig(pasta+'/plots/['+str(datetime.now()).replace(":", "")+'] '+title+".png")
    plt.show()
    return (0,0,0)
def confusion_table1(test_labels, pred, title):
    conf_matrix = metrics.confusion_matrix(test_labels, pred, labels=None) 
    x=[1, 2, 6, 7, 8]
    class_names = ['${:d}$'.format(i) for i in x] 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.xaxis.tick_top()
    major_ticks = range(0,5)
    minor_ticks = [x + 0.5 for x in range(0, 5)]
    ax.xaxis.set_ticks(major_ticks, minor=False)
    ax.yaxis.set_ticks(major_ticks, minor=False)
    ax.xaxis.set_ticks(minor_ticks, minor=True)
    ax.yaxis.set_ticks(minor_ticks, minor=True)
    ax.xaxis.set_ticklabels(class_names, minor=False, fontsize=15)
    ax.yaxis.set_ticklabels(class_names, minor=False, fontsize=15)
    ax.yaxis.set_label_position("right")
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    fig.suptitle(title, y=1.04, fontsize=14, ha='center', fontweight='bold')
    ax.grid(b=True, which=u'minor')
    ax.imshow(conf_matrix, interpolation='nearest', cmap='binary')
    for x in range(conf_matrix.shape[0]):
        for y in range(conf_matrix.shape[1]):
            color = 'w' if x == y else 'k'
            ax.text(x, y, conf_matrix[y,x], ha="center", va="bottom", color=color)       
    pasta=os.path.dirname(__file__)     
    plt.savefig(pasta+'/plots/['+str(datetime.now()).replace(":", "")+'] Confusion table.png', bbox_inches='tight')
    plt.show()
    
def confusion_table2(test_labels, pred, title):
    conf_matrix = metrics.confusion_matrix(test_labels, pred, labels=None)
    x = [3, 4, 5, 9, 10, 11, 12]
    class_names = ['${:d}$'.format(i) for i in x]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.xaxis.tick_top()
    major_ticks = range(0,7)
    minor_ticks = [x + 0.5 for x in range(0, 7)]
    ax.xaxis.set_ticks(major_ticks, minor=False)
    ax.yaxis.set_ticks(major_ticks, minor=False)
    ax.xaxis.set_ticks(minor_ticks, minor=True)
    ax.yaxis.set_ticks(minor_ticks, minor=True)
    ax.xaxis.set_ticklabels(class_names, minor=False, fontsize=15)
    ax.yaxis.set_ticklabels(class_names, minor=False, fontsize=15)
    ax.yaxis.set_label_position("right")
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    fig.suptitle(title, y=1.04, fontsize=14, ha='center', fontweight='bold')
    ax.grid(b=True, which=u'minor')
    ax.imshow(conf_matrix, interpolation='nearest', cmap='binary')
    for x in range(conf_matrix.shape[0]):
        for y in range(conf_matrix.shape[1]):
            color = 'w' if x == y else 'k'
            ax.text(x, y, conf_matrix[y,x], ha="center", va="bottom", color=color)  
    pasta=os.path.dirname(__file__)     
    plt.savefig(pasta+'/plots/['+str(datetime.now()).replace(":", "")+'] Confusion table.png', bbox_inches='tight')
    plt.show()

def confusion_table3(test_labels, pred, title):
    conf_matrix = metrics.confusion_matrix(test_labels, pred, labels=None)
    x = list(range(1, 22))
    class_names = ['${:d}$'.format(i) for i in x]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.xaxis.tick_top()
    major_ticks = range(0,21)
    minor_ticks = [x + 0.5 for x in range(0, 21)]
    ax.xaxis.set_ticks(major_ticks, minor=False)
    ax.yaxis.set_ticks(major_ticks, minor=False)
    ax.xaxis.set_ticks(minor_ticks, minor=True)
    ax.yaxis.set_ticks(minor_ticks, minor=True)
    ax.xaxis.set_ticklabels(class_names, minor=False, fontsize=15)
    ax.yaxis.set_ticklabels(class_names, minor=False, fontsize=15)
    ax.yaxis.set_label_position("right")
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    fig.suptitle(title, y=1.04, fontsize=14, ha='center', fontweight='bold')
    ax.grid(b=True, which=u'minor')
    ax.imshow(conf_matrix, interpolation='nearest', cmap='binary')
    for x in range(conf_matrix.shape[0]):
        for y in range(conf_matrix.shape[1]):
            color = 'w' if x == y else 'k'
            ax.text(x, y, conf_matrix[y,x], ha="center", va="bottom", color=color)       
    pasta=os.path.dirname(__file__)     
    plt.savefig(pasta+'/plots/['+str(datetime.now()).replace(":", "")+'] Confusion table.png', bbox_inches='tight')
    plt.show()

def read_data(error=0, is_train=True):
    pasta=os.path.dirname(__file__)+'\\data\\'
    fi = os.path.join(pasta, 
        ('d0' if error < 10 else 'd') + str(error) + ('_te.dat' if is_train else '.dat'))
    with open(fi, 'r') as fr:
        data = fr.read()
    data = np.fromstring(data, dtype=np.float32, sep='   ')
    if fi == 'data/d00.dat':
        data = data.reshape(-1, 500).T
    else:
        data = data.reshape(-1, 52)
    if is_train:
        data = data[160: ]
    return data, np.ones(data.shape[0], np.int64) * error

def gen_seq_data(target, n_samples, is_train):
    seq_data, seq_labels = [], []
    for i, t in enumerate(target):
        d, _ = read_data(t, is_train)
        data = []
        length = d.shape[0] - n_samples + 1
        for j in range(n_samples):
            data.append(d[j : j + length])
        data = np.hstack(data)
        seq_data.append(data)
        seq_labels.append(np.ones(data.shape[0], np.int64) * i)
    return np.vstack(seq_data), np.concatenate(seq_labels)


def train(model, optimizer, train_loader):
    model.train()
    acc = AccMectric()
    for data, labels in train_loader:
        x = torch.autograd.Variable(data.cpu())
        y = torch.autograd.Variable(labels.cpu())
        o = model(x)        
        loss = torch.nn.NLLLoss()(torch.nn.LogSoftmax()(o), y)
        acc.update(labels.numpy(), o.data.cpu().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return acc.get()


def validate(model, test_loader):
    model.eval()
    acc = AccMectric()
    pred = []
    targets = []
    for data, labels in test_loader:
        x = torch.autograd.Variable(data.cpu())
        o = model(x)
        outputs = o.data.cpu().numpy()
        acc.update(labels.numpy(), outputs)
        pred.extend(outputs.argmax(axis=1))
        targets.extend(labels.numpy())
    return (acc.get(), np.asarray(pred), np.asarray(targets))


class CNN(torch.nn.Module):
    def __init__(self, i, h, o, n):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv1d(1, 8, kernel_size=8),
            torch.nn.BatchNorm1d(8),
            torch.nn.ReLU(),
            torch.nn.Dropout(.2),
            torch.nn.MaxPool1d(2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv1d(8, 16, kernel_size=4),
            torch.nn.BatchNorm1d(16),           
            torch.nn.ReLU(),
            torch.nn.Dropout(.2),
            torch.nn.MaxPool1d(2))
        self.fc = torch.nn.Linear(16*35, h)
        self.b2 = torch.nn.BatchNorm1d(h)
        self.a2 = torch.nn.LeakyReLU(0.01, True)
        self.sm = torch.nn.Linear(h, o)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, 16*35)
        x = self.a2(self.b2(self.fc(x)))
        x = self.sm(x)
        return x

def CNN_TE(n_samples, n_hidden, target, train_data, train_labels, test_data, test_labels):
    train_data = np.expand_dims(train_data, axis=1)
    test_data = np.expand_dims(test_data, axis=1)  
    train_dataset = tchdata.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels))
    test_dataset = tchdata.TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels))
    train_loader = tchdata.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = tchdata.DataLoader(test_dataset, batch_size=32, shuffle=False)    
    model = CNN(52 * n_samples, n_hidden, len(target), n_samples)
    model.cpu()
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.005)
    train_acc_CNN=[]
    test_acc_CNN=[]
    pred = []
    targets = []
    for i in range(60):
        train_acc = train(model, optimizer, train_loader)
        test_acc, pred, targets = validate(model, test_loader)
        train_acc_CNN.append(train_acc)
        test_acc_CNN.append(test_acc)
        print('{}\tépocas = {}\tTreino Acurácia: {:0.3f}\tTeste Acurácia: {:0.3f}' \
            .format(datetime.now(), i, train_acc, test_acc))        
    return (train_acc_CNN, test_acc_CNN, targets, pred)

n_samples = 3      
n_hidden = 30
n_components=30
target1 =[1, 2, 6, 7, 8]#list(range(1, 22)) 
train_data1, train_labels1 = gen_seq_data(target1, n_samples, is_train=True)
test_data1, test_labels1 = gen_seq_data(target1, n_samples, is_train=False)
scaler1 = preprocessing.StandardScaler().fit(train_data1)
train_data1 = scaler1.transform(train_data1)
test_data1 = scaler1.transform(test_data1)
cnn1 = CNN_TE(n_samples, n_hidden, target1, train_data1, train_labels1, test_data1, test_labels1)
train_acc1, test_acc1, targets1, pred1 = cnn1
plots((train_acc1, 'Treino Acurácia'), 'Acurácia', 'Comparação entre Acurácias CNN',  num_iterations=60, x_label='iterations',second=(test_acc1, 'test Acurácia'), third=None, show_min=True)
confusion_table1(targets1, pred1, 'Matriz confusão CNN')


target2 = [3, 4, 5, 9, 10, 11, 12]
train_data2, train_labels2 = gen_seq_data(target2, n_samples, is_train=True)
test_data2, test_labels2 = gen_seq_data(target2, n_samples, is_train=False)
scaler2 = preprocessing.StandardScaler().fit(train_data2)
train_data2 = scaler2.transform(train_data2)
test_data2 = scaler2.transform(test_data2)
cnn2 = CNN_TE(n_samples, n_hidden, target2, train_data2, train_labels2, test_data2, test_labels2)
train_acc2, test_acc2, targets2, pred2 = cnn2
plots((train_acc2, 'Treino Acurácia'), 'Acurácia', 'Comparação entre Acurácias CNN',  num_iterations=60, x_label='iterations',second=(test_acc2, 'test Acurácia'), third=None, show_min=True)
confusion_table2(targets2, pred2, 'Matriz confusão CNN')


target3 = list(range(1, 22)) 
train_data3, train_labels3 = gen_seq_data(target3, n_samples, is_train=True)
test_data3, test_labels3 = gen_seq_data(target3, n_samples, is_train=False)
scaler3 = preprocessing.StandardScaler().fit(train_data3)
train_data3 = scaler3.transform(train_data3)
test_data3 = scaler3.transform(test_data3)
cnn3 = CNN_TE(n_samples, n_hidden, target3, train_data3, train_labels3, test_data3, test_labels3)
train_acc3, test_acc3, targets3, pred3 = cnn3
plots((train_acc3, 'Treino Acurácia'), 'Acurácia', 'Comparação entre Acurácias CNN',  num_iterations=60, x_label='iterations',second=(test_acc3, 'test Acurácia'), third=None, show_min=True)
confusion_table3(targets3, pred3, 'Matriz confusão CNN')

