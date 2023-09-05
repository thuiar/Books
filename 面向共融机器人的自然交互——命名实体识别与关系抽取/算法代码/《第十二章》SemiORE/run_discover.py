from relation_discover.config import Param
from relation_discover.dataloaders.data_field import *
from relation_discover.methods.manager import Manager

def run(args):
    if args.dataname in ['fewrel']:
        args.n_clusters = 16
    elif args.dataname in ['cpr']:
        args.n_clusters = 5
    elif args.dataname in ['fewrel2.0']:
        args.n_clusters = 10
    args.pretrain_name = args.dataname
    logger.info(str(args.__dict__))
    data = Data(args)
    manager = Manager(args)   
    
    logger.info('Training Begin...')
    manager.train(args, data)
    logger.info('Training Finished...')
    logger.info('Evaluation begin...')
    manager.restore_model(args)
    results = manager.eval(args, data.test_data_loader)
    save_results(args, results)

    logger.info('Evaluation finished...')

if __name__ == '__main__':
    torch.cuda.set_device(1)
    param = Param()
    args = param.args
    # args.loop_nums = 1
    # args.pre_loop_nums = 1
    run(args)
    

   