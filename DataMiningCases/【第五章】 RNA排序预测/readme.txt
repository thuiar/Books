文本挖掘和图像挖掘文件夹包含：
1）	原始数据
文本挖掘工作的原始数据为7个excel表格；图像挖掘工作的原始数据为60张jpg图片和记录人工挑选颗粒坐标的star文件。
2）	预处理数据与预处理程序
文本挖掘工作的预处理数据为txt文件和由python的cPickle包处理过的存储文件；图像挖掘工作的预处理文件为60张经过处理的jpg图片和由python的cPickle包处理过的存储文件以及matlab可识别的mat文件。（以上均为python语言）
3）	模型学习程序与数据可视化
文本挖掘工作的模型包含PCA+SVM、LDA+SVM、各种模型程序、CNN（均为python语言）等；图像挖掘工作的模型包含CNN、各种模型程序、SVM（均为python语言）、神经网络、CNN（matlab语言）。数据可视化包含在以上挖掘工作程序中。


注：因为网络学堂要求不能上传过大文件，这里将预处理之后规整的数据集（mat,pkl,npy文件）全部删除，没有上传（因为都非常大，600MB到2G不等）