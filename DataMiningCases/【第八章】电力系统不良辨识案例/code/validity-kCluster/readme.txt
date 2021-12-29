程序说明：
calDist2M.m：		求某数据列X与质心列C的距离平方和，返回一个矩阵
updateC.m：		更新质心位置
updateClass.m：		更新聚类
kClusterMain.m：	主函数
deleteoutliers.m：	用Grubbs检验剔除不良数据

数据说明：
MeterData_lxz.mat：	见cCluster文件夹
MeterData_lxz_K(kC=3)：	存放聚类结果（聚类个数为3，不良数据个数为100）