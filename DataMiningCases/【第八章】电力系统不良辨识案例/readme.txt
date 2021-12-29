GSA-kCluster：		基于GSA的k-means聚类算法程序
validity-kCluster：	基于有效指数的k-means聚类算法程序
cCluster：		模糊C均值聚类算法程序

readData.m：		从.csv文件中读取所需数据（该文件较大，已经删除）

程序中用到的数据集在cCluster文件夹的MeterData_lxz.mat中，原始数据文件较大（约200M），因此没有放入文件夹中

MeterData_lxz.mat：	存放从.csv文件中读取的数据，主要数据存于7维矩阵DataM中，各维度含义见readData.m