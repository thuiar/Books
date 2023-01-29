import os
import random
import argparse

__all__ = ['Config', 'ConfigDebug']

class Storage(dict):
    """
    A Storage object is like a dictionary except `obj.foo` can be used inadition to `obj['foo']`
    ref: https://blog.csdn.net/a200822146085/article/details/88430450
    """
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __str__(self):
        return "<" + self.__class__.__name__ + dict.__repr__(self) + ">"

class Config():
    def __init__(self, input_args):
        # global parameters for running
        try:
            self.global_running = vars(input_args)
        except TypeError:
            self.global_running = input_args
        # hyper parameters for models
        self.HYPER_MODEL_MAP = {
            'FER_DCNN': self.__FER_DCNN,
            'LM_DCNN': self.__LM_DCNN,
            'CMCNN': self.__CMCNN,
        }
        # hyper parameters for datasets
        self.HYPER_DATASET_MAP = self.__datasetCommonParams()
    
    def __datasetCommonParams(self):
        data_root_dir = '/home/sharing/disk3/dataset/facial-expression-recognition'
        tmp = {
            'RAF':{
                'data_dir': os.path.join(data_root_dir, 'RAF-Basic/Processed/Faces'),
                'label_dir': os.path.join(data_root_dir, 'RAF-Basic/Processed/Label'),
                'fer_num_classes': 7,
                # 'fer_num_classes': 6,
                'num_landmarks': 68,
                'num_tasks': 2,
            },
            'SFEW2':{
                'data_dir': os.path.join(data_root_dir, 'SFEW2/Processed/AlignedFaces'),
                'label_dir': os.path.join(data_root_dir, 'SFEW2/Processed/AlignedLabels'),
                'fer_num_classes': 7,
                'num_landmarks': 68,
                'num_tasks': 2,
            },
            'CK+':{
                'data_dir': os.path.join(data_root_dir, 'CK+/Processed/AlignedFaces'),
                'label_dir': os.path.join(data_root_dir, 'CK+/Processed/AlignedLabels'),
                'fer_num_classes': 7,
                # 'fer_num_classes': 6,
                'num_landmarks': 68,
                'num_tasks': 2,
            },
            'MMI':{
                'data_dir': os.path.join(data_root_dir, 'MMI/Processed/AlignedFaces'),
                'label_dir': os.path.join(data_root_dir, 'MMI/Processed/AlignedLabels'),
                'fer_num_classes': 6,
                'num_landmarks': 68,
                'num_tasks': 2,
            },
            'OULU_CASIA':{
                'data_dir': os.path.join(data_root_dir, 'Oulu_CASIA/Processed/AlignedFaces'),
                'label_dir': os.path.join(data_root_dir, 'Oulu_CASIA/Processed/AlignedLabels'),
                'fer_num_classes': 6,
                'num_landmarks': 68,
                'num_tasks': 2,
            },
        }
        return tmp
   
    def __FER_DCNN(self):
        tmp = {
            'commonParas':{
                # Tuning
                'metricsName': 'FER',
                'epochs': 50,
                'patience': 15, # when to decay learning rate
                # Logistics
                'keyEval': 'Average_Acc',
            },
            # dataset
            'datasetParas':{
                'RAF':{
                    'batch_size': 32,
                    'embedding_size': 128,
                    'weight_decay': 0.005,
                    'lr': 0.01,
                },
                'SFEW2':{
                    'batch_size': 128,
                    'embedding_size': 128,
                    'weight_decay': 0.0001,
                    'lr': 0.01,
                },
                'CK+':{
                    'batch_size': 32,
                    'embedding_size': 128,
                    'weight_decay': 0.005,
                    'lr': 0.01,
                },
                'MMI':{
                    'batch_size': 16,
                    'embedding_size': 256,
                    'weight_decay': 0.005,
                    'lr': 0.01,
                },
                'OULU_CASIA':{
                    'batch_size': 16,
                    'embedding_size': 256,
                    'weight_decay': 0.005,
                    'lr': 0.01,
                },
            },
        }
        return tmp
    
    def __LM_DCNN(self):
        tmp = {
            'commonParas':{
                # Tuning
                'metricsName': 'LM',
                'epochs': 50,
                'patience': 15, # when to decay learning rate
                # Logistics
                'keyEval': 'NME',
            },
            # dataset
            'datasetParas':{
                'RAF':{
                    'batch_size': 32,
                    'embedding_size': 2048,
                    'weight_decay': 5e-3,
                    'learning_rate': 0.01,
                },
                'SFEW2':{
                    'batch_size': 32,
                    'embedding_size': 1024,
                    'weight_decay': 0.05,
                    'learning_rate': 0.1,
                },                
                'CK+':{
                    'batch_size': 8,
                    'embedding_size': 1024,
                    'weight_decay': 0.01,
                    'learning_rate': 0.001,
                },
                'MMI':{
                    'batch_size': 8,
                    'embedding_size': 1024,
                    'weight_decay': 0.05,
                    'learning_rate': 0.001,
                },
                'OULU_CASIA':{
                    'batch_size': 8,
                    'embedding_size': 512,
                    'weight_decay': 0.05,
                    'learning_rate': 0.001,
                },
            },
        }
        return tmp

    def __CMCNN(self):
        tmp = {
            'commonParas':{
                # Tuning
                'metricsName': 'FER_LM',
                'epochs': 50,
                'patience': 15, # when to decay learning rate
                # Logistics
                'keyEval': 'Average_Acc',
            },
            # dataset
            'datasetParas':{
                'RAF':{
                    'batch_size': 32,
                    'e_ratio': 0.2, 
                    'fer_embedding': 512,
                    'lm_embedding': 512,
                    'alphaBetas': [[0.5, 0.5]],
                    'lm_threshold': 0.5,
                    'lambda_e2l': 0.2,
                    'lambda_l2e': 1.0,
                    # 'alphaBetas': [[0.0, 1.0]],
                    'weight_decay': 0.005,
                    'loss_fer': 1.0,
                    'loss_lm': 0.5,
                    'loss_mtl': 1.0,
                    'lr': 0.01,
                },
                'SFEW2':{
                    'batch_size': 32,
                    'e_ratio': 0.2, 
                    'fer_embedding': 128,
                    'lm_embedding': 2048,
                    'alphaBetas': [[0.5, 0.5]],
                    'lm_threshold': 0.5,
                    'lambda_e2l': 0.2,
                    'lambda_l2e': 0.2,
                    # 'alphaBetas': [[0.0, 1.0]],
                    'weight_decay': 0.005,
                    'loss_fer': 1.0,
                    'loss_lm': 0.5,
                    'loss_mtl': 1.0,
                    'lr': 0.01,
                },
                'CK+':{
                    'batch_size': 16,
                    'e_ratio': 0.5, 
                    'fer_embedding': 128,
                    'lm_embedding': 2048,
                    'alphaBetas': [[0.5, 0.5]],
                    'lm_threshold': 0.5,
                    'lambda_e2l': 0.2,
                    'lambda_l2e': 0.2,
                    # 'alphaBetas': [[0.0, 1.0]],
                    'weight_decay': 0.005,
                    'loss_fer': 1.0,
                    'loss_lm': 0.5,
                    'loss_mtl': 1.0,
                    'lr': 0.01,
                },
                'MMI':{
                    'batch_size': 32,
                    'e_ratio': 0.5, 
                    'fer_embedding': 128,
                    'lm_embedding': 2048,
                    'alphaBetas': [[0.5, 0.5]],
                    'lm_threshold': 0.5,
                    'lambda_e2l': 0.2,
                    'lambda_l2e': 0.2,
                    # 'alphaBetas': [[0.0, 1.0]],
                    'weight_decay': 0.0001,
                    'loss_fer': 1.0,
                    'loss_lm': 0.5,
                    'loss_mtl': 1.0,
                    'lr': 0.01,
                },
                'OULU_CASIA':{
                    'batch_size': 32,
                    'e_ratio': 0.5, 
                    'fer_embedding': 128,
                    'lm_embedding': 2048,
                    'alphaBetas': [[0.5, 0.5]],
                    'lm_threshold': 0.5,
                    'lambda_e2l': 0.2,
                    'lambda_l2e': 0.2,
                    # 'alphaBetas': [[0.0, 1.0]],
                    'weight_decay': 0.0001,
                    'loss_fer': 1.0,
                    'loss_lm': 0.5,
                    'loss_mtl': 1.0,
                    'lr': 0.01,
                },
            },
        }
        return tmp

    def get_config(self):
        # normalize
        model_name = self.global_running['modelName'].upper()
        dataset_name = self.global_running['datasetName'].upper()
        # integrate all parameters
        res =  Storage(dict(self.global_running,
                            **self.HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                            **self.HYPER_MODEL_MAP[model_name]()['commonParas'],
                            **self.HYPER_DATASET_MAP[dataset_name]))
        return res


class ConfigDebug():
    def __init__(self, input_args):
        # global parameters for running
        try:
            self.global_running = vars(input_args)
        except TypeError:
            self.global_running = input_args
        # hyper parameters for models
        self.HYPER_MODEL_MAP = {
            'FER_DCNN': self.__FER_DCNN,
            'LM_DCNN': self.__LM_DCNN,
            'CMCNN': self.__CMCNN
        }
        # hyper parameters for datasets
        self.HYPER_DATASET_MAP = self.__datasetCommonParams()
    
    def __datasetCommonParams(self):
        data_root_dir = '/home/sharing/disk3/dataset/facial-expression-recognition'
        tmp = {
            'RAF':{
                'data_dir': os.path.join(data_root_dir, 'RAF-Basic/Processed/Faces'),
                'label_dir': os.path.join(data_root_dir, 'RAF-Basic/Processed/Label'),
                'fer_num_classes': 7,
                'num_landmarks': 68,
            },
            'SFEW':{
                'data_dir': os.path.join(data_root_dir, 'SFEW2/Processed/AlignedFaces'),
                'label_dir': os.path.join(data_root_dir, 'SFEW2/Processed/Label'),
                'fer_num_classes': 7,
                'num_landmarks': 68,
            },
            'CK+':{
                'data_dir': os.path.join(data_root_dir, 'CK+/Processed/AlignedFacesEqual'),
                'label_dir': os.path.join(data_root_dir, 'CK+/Processed/Label'),
                'fer_num_classes': 7,
                'num_landmarks': 68,
            },
            'MMI':{
                'data_dir': os.path.join(data_root_dir, 'MMI/Processed/AlignedFacesEqual'),
                'label_dir': os.path.join(data_root_dir, 'MMI/Processed/Label'),
                'fer_num_classes': 6,
                'num_landmarks': 68,
            },
            'OULU_CASIA':{
                'data_dir': os.path.join(data_root_dir, 'Oulu_CASIA/Processed/AlignedFacesEqual'),
                'label_dir': os.path.join(data_root_dir, 'Oulu_CASIA/Processed/Label'),
                'fer_num_classes': 6,
                'num_landmarks': 68,
            },

        }
        return tmp
   
    def __FER_DCNN(self):
        tmp = {
            'commonParas':{
                # Tuning
                'metricsName': 'FER',
                'epochs': 50,
                'patience': 15, # when to decay learning rate
                # Logistics
                'keyEval': 'Average_Acc',
            },
            # dataset
            'datasetParas':{
                'RAF':{
                    # ref Original Paper
                    'd_paras': ['batch_size', 'embedding_size', 'weight_decay', 'lambda_islandLoss', 'weight_islandLoss', 'lr'],
                    'batch_size': random.choice([128,64]),
                    'embedding_size': random.choice([128,256]),
                    'weight_decay': random.choice([0.1,0.005,0.0]),
                    'lambda_islandLoss': random.choice([1,10,20]),
                    'weight_islandLoss': random.choice([0.1,0.01]),
                    'lr': random.choice([0.01]),
                    'loss_lr': random.choice([0.01]),
                },
                'CK+':{
                    'd_paras': ['batch_size', 'embedding_size', 'weight_decay', 'lambda_islandLoss', 'weight_islandLoss', 'lr'],
                    'batch_size': random.choice([64,32,16]),
                    'embedding_size': random.choice([128]),
                    'weight_decay': random.choice([0.005]),
                    'lambda_islandLoss': random.choice([20]),
                    'weight_islandLoss': random.choice([0.1]),
                    'lr': random.choice([0.01]),
                    'loss_lr': random.choice([0.01]),
                },
                'OULU_CASIA':{
                    'd_paras': ['batch_size', 'embedding_size', 'weight_decay', 'lambda_islandLoss', 'weight_islandLoss', 'lr'],
                    'batch_size': random.choice([8,16]),
                    'embedding_size': random.choice([128,256]),
                    'weight_decay': random.choice([0.1,0.01,0.005,0.0]),
                    'lambda_islandLoss': random.choice([1,10,20]),
                    'weight_islandLoss': random.choice([0.1,0.01]),
                    'lr': random.choice([0.01]),
                    'loss_lr': random.choice([0.01]),
                },
            },
        }
        return tmp

    def __LM_DCNN(self):
        tmp = {
            'commonParas':{
                # Tuning
                'metricsName': 'LM',
                'epochs': 100,
                'patience': 20, # when to decay learning rate
                # Logistics
                'keyEval': 'NME',
            },
            # dataset
            'datasetParas':{
                'RAF':{
                    # ref Original Paper
                    'd_paras': ['batch_size', 'embedding_size', 'weight_decay', 'learning_rate'],
                    'batch_size': random.choice([128,64]),
                    'embedding_size': random.choice([512, 1024, 2048]),
                    'weight_decay': random.choice([0.02,0.1,0.05,0.01,0.005]),
                    'learning_rate': random.choice([0.1, 0.01, 0.001, 0.0001]),
                },
                'SFEW':{
                    # ref Original Paper
                    'd_paras': ['batch_size', 'embedding_size', 'weight_decay', 'learning_rate'],
                    'batch_size': random.choice([64,32,16,8]),
                    'embedding_size': random.choice([512, 1024, 2048]),
                    'weight_decay': random.choice([0.02,0.1,0.05,0.01,0.005]),
                    'learning_rate': random.choice([0.1, 0.01, 0.001, 0.0001]),
                },
                'CK+':{
                    # ref Original Paper
                    'd_paras': ['batch_size', 'embedding_size', 'weight_decay', 'learning_rate'],
                    'batch_size': random.choice([64,32,16,8]),
                    'embedding_size': random.choice([512, 1024, 2048]),
                    'weight_decay': random.choice([0.02,0.1,0.05,0.01,0.005]),
                    'learning_rate': random.choice([0.1, 0.01, 0.001, 0.0001]),
                },
                'OULU_CASIA':{
                    # ref Original Paper
                    'd_paras': ['batch_size', 'embedding_size', 'weight_decay', 'learning_rate'],
                    'batch_size': random.choice([64,32,16,8]),
                    'embedding_size': random.choice([512, 1024, 2048]),
                    'weight_decay': random.choice([0.02,0.1,0.05,0.01,0.005]),
                    'learning_rate': random.choice([0.1, 0.01, 0.001, 0.0001]),
                },
            },
        }
        return tmp
   
    def __CMCNN(self):
        tmp = {
            'commonParas':{
                # Tuning
                'metricsName': 'FER_LM',
                'epochs': 22,
                'patience': 10, # when to decay learning rate
                # Logistics
                'keyEval': 'Average_Acc',
            },
            # dataset
            'datasetParas':{
                'RAF':{
                    'batch_size': 32,
                    'e_ratio': 0.5, 
                    'lr': 0.01,
                    'fer_embedding': 128,
                    'lm_embedding': 512,
                    'alphaBetas': [[0.5, 0.5]],
                    'weight_decay': 0.005,
                    'lm_threshold': 0.5,
                    'd_paras': ['lambda_e2l', 'lambda_l2e', 'loss_fer', \
                                'loss_lm', 'loss_mtl'],
                    'lambda_e2l': random.choice([0.1, 1.0]),
                    'lambda_l2e': random.choice([0.1, 1.0]),
                    'loss_fer': random.choice([1.0, 0.5]),
                    'loss_lm': random.choice([1.0, 0.5]),
                    'loss_mtl': random.choice([1.0, 0.1, 0.01]),
                },
                'CK+':{
                    'd_paras': ['batch_size', 'e_ratio', 'fer_embedding', 'lm_embedding', \
                                'alphaBetas', 'weight_decay', 'weight_lm', 'weight_lp', 'lr'],
                    'batch_size': random.choice([16,32]),
                    'e_ratio': random.choice([0.2, 0.5]),
                    'fer_embedding': random.choice([128]),
                    'lm_embedding': random.choice([2048]),
                    'alphaBetas': random.choice([[[0.5, 0.5]], [[1.0, 0.0]], [[0.0, 1.0]], [[0.2, 0.8]], [[0.8, 0.2]]]),
                    'weight_decay': random.choice([0.005]),
                    'weight_lm': random.choice([0.5]),
                    'weight_lp': random.choice([0.01, 0.1, 1]),
                    'lr': random.choice([0.01]),
                },
                'OULU_CASIA':{
                    'd_paras': ['batch_size', 'e_ratio', 'fer_embedding', 'lm_embedding', \
                                'alphaBetas', 'weight_decay', 'weight_lm', 'weight_lp', 'lr'],
                    'batch_size': random.choice([16,32]),
                    'e_ratio': random.choice([0.2, 0.5]),
                    'fer_embedding': random.choice([128]),
                    'lm_embedding': random.choice([1024]),
                    'alphaBetas': random.choice([[[0.5, 0.5]], [[1.0, 0.0]], [[0.0, 1.0]], [[0.2, 0.8]], [[0.8, 0.2]]]),
                    'weight_decay': random.choice([0.005]),
                    'weight_lm': random.choice([0.1, 0.5]),
                    'weight_lp': random.choice([0.01, 0.1, 1]),
                    'lr': random.choice([0.01]),
                }
           },
        }
        return tmp
    
    def get_config(self):
        # normalize
        model_name = self.global_running['modelName'].upper()
        dataset_name = self.global_running['datasetName'].upper()
        # integrate all parameters
        res =  Storage(dict(self.global_running,
                            **self.HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                            **self.HYPER_MODEL_MAP[model_name]()['commonParas'],
                            **self.HYPER_DATASET_MAP[dataset_name]))
        return res