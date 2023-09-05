[![](https://badgen.net/badge/license/GPL-3.0/green)](#License)
# TEXTORE: An Integrated Visualization Platform for Text Open Relation Extraction
TEXTORE is the first integrated visualization platform for text open relation extraction. First, it proposes a new task, the detection of open relations, and integrates the discovery of open relations. In addition to this, it contains a pipeline framework to perform open relation detection and open relation discovery simultaneously. Demonstrate the entire process and pipeline framework of the two submodules by building a visualization system.




# A Pipeline Framework for Open Relation Extracion
We propose a new task of open relation detection and go through the whole process of identifying known relations and discovering open relations through a pipeline scheme. 

First, we prepare commonly used data for two modules (open relation detection and open relation discovery) and set the labeled ratio and known relation ratio. 

Then, after the data and model are ready, we use the labeled known relation data to train the selected open relation detection method. A trained detection model is used to predict known relations and detect open relations. Then, an open relation discovery model is trained using the predicted and labeled known relation. The trained discovery model further clusters open relations into fine-grained relation classes.

Finally, we combine the prediction results of both modules, containing the identified known relations and the discovered clusters of open relations.

<div align="center">
<img src=figs/overframe.png width=80% />
</div> 

# Visualized Platform
The visualized platform contains four parts, data management, open relation detection, open relation discovery and open realtion  extraction. 

The TEXTORE visualized platform:

<div align="center">
<img src=figs/view.png width=80% />
</div> 


#   Tutorials
## a.  Environmental Installation  
Use anaconda to create Python (version >= 3.8) environment:
        
    conda create --name yourname python=3.8
    conda activate yourname

Install PyTorch :
    
    pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

Install related environmental dependencies:
    
    cd ORE
    pip install -r requirements.txt

    

## b. Quick Start
    
You can choose the port number  in  manage.py  then  execute code:
            
    python manage.py 


#   System Basic Information
##  Benchmark Datasets
We integate four benchmark relation extraction datasets as follows:
* [NYT10m](https://aclanthology.org/2021.findings-acl.112.pdf)
* [Wiki20](https://aclanthology.org/2021.findings-acl.112.pdf)
* [Wiki80m](https://aclanthology.org/D19-3029.pdf)
* [SemEval](https://aclanthology.org/S10-1006.pdf)

### Dataset Display
The visual interface is used for the basic information of the tactical dataset, such as the introduction of the dataset, the number of categories, and the source.

<div align="center">
<img src=figs/data-show.png width=80% />
</div>

Click through Operations to view the details of the methods.

<div align="center">
<img src=figs/data-info.png width=80% />
</div>

### Add a Dataset
This visual interface is convenient for adding new data sets and editing their related information.
We should prepare train.json, dev.json and test.json (hardly consistent with the data format) and upload them in the add interface.

<div align="center">
<img src=figs/add-data.png width=80% />
</div>

###  Data Settings
Each dataset is divided into training set(labeled and unlabeled) , development set and test set. We choose partial relations as known (marking ratios can be changed) relations. It is worth noting that we uniformly select 10% from the known relational data as the mark. We use all training data to train the model. During the test, we evaluate the clustering performance of all relation classes. More detailed information can be seen in the paper.


## Open Relation Detection
This module integrates five advanced open  detection methods:

* DOC: Deep Open Classification of Text Documents(EMNLP,2017) [[paper]](https://www.aclweb.org/anthology/D17-1314.pdf)[[code]](https://github.com/leishu02/EMNLP2017_DOC)

* A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks(ICLR,2017)[[paper]](https://arxiv.org/pdf/1610.02136.pdf)[[code]](https://github.com/facebookresearch/odin)

* Towards Open Set Deep Networks(CVPR,2016)[[paper]](https://openaccess.thecvf.com/content_cvpr_2016/papers/Bendale_Towards_Open_Set_CVPR_2016_paper.pdf)[[code]](https://github.com/thuiar/TEXTOIR/tree/main/open_intent_detection/methods/OpenMax)

* Deep Unknown Intent Detection with Margin Loss(ACL,2019)[[paper]](https://aclanthology.org/P19-1548.pdf)[[code]](https://github.com/thuiar/DeepUnkID)

* Deep Open Intent Classification with Adaptive Decision Boundary(AAAI,2019)[[paper]](https://arxiv.org/pdf/2012.10209.pdf)[[code]](https://github.com/thuiar/Adaptive-Decision-Boundary)

### Model Management
We integrated the latest unknown detection methods and migrated them to the field of relation extraction, and displayed basic information such as method types and provenance papers on this interface.

<div align="center">
<img src=figs/model-info.png width=80% />
</div>


You can download the code (code address) of this platform, and change and run your method according to our display logic, or you can contact us and we will integrate it into the platform.
### Model Training
In this interface, users can add new training records by assigning methods, data sets, known relation ratios, and labeling ratios. In addition, set the hyperparameters when choosing different methods. As shown below:

<div align="center">
<img src=figs/model1-train.png width=80% />
</div>

During the training, the user can observe the training status (completed or failed) of the task started through the interface, and view the basic information of the task setting. After the training is completed, the user can jump to the corresponding evaluation and analysis module of the task.

<div align="center">
<img src=figs/model1-train1.png width=80% />
</div>

### Model Evaluation
 Users can view the corresponding evaluation indicators and image display by selecting the data set, detection method, creation time, etc., such as the change of loss during the training process, as shown in the following figure:

<div align="center">
<img src=figs/dete-eval1.png width=80% />
</div>

<div align="center">
<img src=figs/dete-eval2.png width=80% />
</div>

<div align="center">
<img src=figs/dete-eval3.png width=80% />
</div>

### Model Analysis
Similar to model evaluation, users can view the corresponding analysis image display by selecting the data set, detection method, creation time, etc., as shown in the following figure:

<div align="center">
<img src=figs/dete-ana1.png width=80% />
</div>

<div align="center">
<img src=figs/dete-ana2.png width=80% />
</div>

## Open Relation Discovery
This module integrates six advanced open  discovery methods:

* ODC: Online Deep Clustering for Unsupervised Representation Learning. (CVPR 2020)[[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhan_Online_Deep_Clustering_for_Unsupervised_Representation_Learning_CVPR_2020_paper.pdf)[[code]](https://github.com/open-mmlab/OpenSelfSup)
    
* SelfORE: Self-ORE: Self-supervised Relational Feature Learning for Open Relation Extraction. (EMNLP2020)[[paper]](https://aclanthology.org/2020.emnlp-main.299.pdf)[[code]](https://github.com/THU-BPM/SelfORE)

* RSN: Open relation extraction: Relational knowledge transfer from supervised data to unsupervised data. (EMNLP 2019) [[paper]](https://aclanthology.org/D19-1021.pdf)[[code]](https://github.com/thunlp/RSN)   

* MORE: More: A Metric Learning Based Framework for Open-Domain Relation Extraction. (ICASSP 2021)[[paper]](https://ieeexplore.ieee.org/document/9413437)[[code]](https://github.com/RenzeLou/MORE)

* Discovering New Intents with Deep Aligned Clustering.[[paper]](https://arxiv.org/pdf/2012.08987.pdf)[[code]](https://github.com/thuiar/DeepAligned-Clustering)

* Discovering New Intents via Constrained Deep Adaptive Clustering with Cluster Refinement[[paper]](https://arxiv.org/abs/1911.08891)[[code]](https://github.com/thuiar/CDAC-plus)

### Model Management
Similar to relation detection, we integrate the latest methods discovered in unknown classes in this module, and divide them into two categories: unsupervised methods and semi-supervised methods.Tthis interface displayed those methods basic information such as method types and provenance papers.

<div align="center">
<img src=figs/dis-eva1.png width=80% />
</div>

Similar to model evaluation,you can add it yourself or contact us for integration.
### Model Training
Similar to the detection interface, in this interface, users add new training records by selecting relevant parameters. As follows:

<div align="center">
<img src=figs/dis-train.png width=80% />
</div>

During the training process, the user can observe the training status (completed or failed) of the starting task through the interface, and view the basic information of the task setting. After the training is completed, the user can jump to the evaluation analysis module corresponding to the task.

### Model Evaluation
Users can view the corresponding evaluation indicators and image display through the parameter list, such as the change of loss during training, as shown in the following figure:

<div align="center">
<img src=figs/disc-eva1.png width=80% />
</div>

### Model Analysis
Similar to model evaluation, users can view the corresponding analysis image display by selecting parameters, etc. As shown in the following figure:

<div align="center">
<img src=figs/disco-eva2.png width=80% />
</div>

## Open Relation Extraction.
### Configuration Parameter
First select a benchmark dataset, then select the label ratio and the known relation ratio, and finally select the detection method and the discovery method to run.

<div align="center">
<img src=figs/pip.png width=80% />
</div>

### Display Results
After running, you can see the known relation samples, open relation samples of the selected dataset, and the keywords, confidence, and quantity of the relations found.

Detecion  Results

<div align="center">
<img src=figs/pip1.png width=80% />
</div>

<div align="center">
<img src=figs/pip2.png width=80% />
</div>

Discovery Results

<div align="center">
<img src=figs/pip3.png width=80% />
</div>


# Acknowledgements

[https://github.com/thuiar/TEXTOIR](https://github.com/thuiar/TEXTOIR)

[https://github.com/thuiar/TEXTOIR-DEMO](https://github.com/thuiar/TEXTOIR-DEMO)


#### Contact
If you have any questions, please contact zhaok7878@gmail.com












  




