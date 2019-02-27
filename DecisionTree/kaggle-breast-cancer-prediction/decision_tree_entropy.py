from __future__ import division
import numpy as np # linear algebra
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
import logging,sys

FORMAT = '%(asctime)-20s [%(levelname)-4s] %(message)s'   #%(asctime)s: 打印日志的时间   %(levelname)s: 打印日志级别名称  %(message)s: 打印日志信息
logging.basicConfig(stream=sys.stdout,format=FORMAT, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%I')  #logging.basicConfig函数对日志的输出格式及方式做相关配置。stream:用指定的stream创建StreamHandler，可以指定输出到sys.stderr,sys.stdout或者文件
#read data from dataset                         format:日志显示格式。 datefmt：指定日期时间格式。 level：设置日志级别 
logging.info("*** CLEANING DATAFRAME ***")
data_frame = pd.read_csv("dataset.csv",header=1)#以第一行数据做列标题，第一行数据就不存在于表中了，表中的第一行数据变为原表第二行数据
data_frame.drop(data_frame.columns[[0]], axis=1, inplace=True)# inplace=True：不创建新的对象，直接对原始对象进行修改；inplace=False：对数据进行修改，创建并返回新的对象承载其修改结果。
dataset = shuffle(np.array(data_frame))#array函数创建数组,shuffle() 方法将序列的所有元素随机排序
print(dataset)

extracted_dataset= []
target = []

#extract target column
for row in dataset:
    extracted_dataset.append(row[1:])
    if row[0] == 'B':
        target.append(0)
    else:
        target.append(1)



X_train, X_test, Y_train, Y_test= train_test_split(extracted_dataset,target,test_size=0.3)
logging.info("*** DATASET PARTITIONED IN TRAIN: "+str(len(X_train))+ " TEST: "+str(len(X_test)))


clf_entropy = DecisionTreeClassifier(criterion = "entropy", max_depth = 50)#criterion:特征选择标准,可以使用"gini"或者"entropy"，前者代表基尼系数，后者代表信息增益    max_depth :决策树最大深
clf_entropy .fit(X_train,Y_train)

logging.info("*** TRAINING END ***")

predicted_entropy = clf_entropy .predict(X_test)


idx = 0
true = 0
false = 0
for i in X_test:
    #logging.info("*** Pred:"+str(predicted[idx])+" real: "+str(Y_test[idx])+" res "+str(predicted[idx]==Y_test[idx])+" ***")

    if predicted_entropy[idx]==Y_test[idx]:
        true +=1
    else:
        false +=1
    idx +=1

accuracy =  (true/(true+false))*100
logging.info("Positive Class: "+str(true))
logging.info("Negative Class: "+str(false))
logging.info("Accuracy Entropy: "+str(accuracy))
