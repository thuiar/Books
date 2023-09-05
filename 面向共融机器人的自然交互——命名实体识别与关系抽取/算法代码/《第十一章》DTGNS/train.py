from dataloader import *
from manager import ModelManager
from config import Param
import importlib

def run(args):
    print(args.method)
    print('Data Preparation...')
    data = Data(args)
    model = importlib.import_module('methods.' + args.method + '.model')
    model = model.Model(args, data.num_labels)
    manager = ModelManager(args, model, data)   

    print('Training Begin...')
    manager.train(args, data)
    print('Training Finished...')
    manager.restore_model(args)
    print('Evaluation begin...')
    manager.evaluation(args, data, mode='test')
    print('Evaluation finished...')

    manager.save_results(args)
    print('Open Relation Classification Finished...')
    
if __name__ == '__main__':
    param = Param()
    args = param.parser.parse_args()
    print(args.__dict__)
    run(args)