from tools.utils import *
from tools.config import Param
from tools.datatools.detection_dataloader import Data as DetData
from tools.datatools.discover_dataloader import Data as DisData
from pipline.exe_pipline import run_pipline

def test_by_pkl(args, data):
    path = args.res_path
    res = load_pickle(path)
    pred = res['pred']
    label = data.test_label_ids
    results = clustering_score(label, pred)
    results['B3'] = results['B3']['F1']
    saver = SaveData(args, save_cfg=False)
    args.this_name = 'by_pkl'
    saver.save_results(args, results, use_thisname=True)

def run(args, go_test=False):
    args.logger.info(args.method)
    args.logger.info('Data Preparation...')
    args.logger.info('task type: {}'.format(args.task_type))
    
    if args.task_type in ['relation_detection']:
        data = DetData(args)
    elif args.task_type in ['relation_discovery']:
        data = DisData(args)

    manager = get_manager(args, data)
    
    if args.train_model:
        if not go_test:
            args.logger.info('Training Begin...')
            manager.train(args, data)
            args.logger.info('Training Finished...')
    args.logger.info('Evaluation begin...')
    # args.load_ckpt = '/home/sharing/disk1/zk/weight_cache/relation_discover/MORE/ckpt_semeval--0-2e-05.pth'
    args.train_model = 0
    if args.test_by_pkl == 0:
        manager.restore_model(args)
        manager.eval(args, data, is_test=True)
    else:
        test_by_pkl(args, data)

    
if __name__ == '__main__':
    print('Parameters Initialization...')
    args = Param()
    go_test = False
    if args.is_pipe == 0:
        
        # if args.dataname in ["semeval"] and args.seed in [0] and args.known_cls_ratio in [0.25]:
        #     go_test = True
        torch.cuda.set_device(args.gpu_id)
        run(args, go_test)
    else:

        run_pipline(args)

            
            
