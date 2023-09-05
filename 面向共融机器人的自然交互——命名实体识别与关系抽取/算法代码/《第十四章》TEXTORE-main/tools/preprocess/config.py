import argparse
import os
from tools.utils import load_yaml, creat_check_path, get_logger

class Param:
    def __init__(self):
        self.base_param = 'tools/preprocess/base_params.yaml'
        args = self.get_base_param()
        # get logger
        mid_dir = creat_check_path(args.output_path, 'processed', args.dataname, "logs")
        self.logger = get_logger(mid_dir, 'processed')

        for k, v in args.__dict__.items():
            setattr(self, k, v)
        
        self.generate_all_path(args)

        self.logger.info("Use config:")
        for k, v in self.__dict__.items():
            if not k.startswith("__") and type(v) in [float, str, int]:
                self.logger.info("{}---{}".format(k, getattr(self, k)))

    def get_base_param(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        d = load_yaml(self.base_param)
        for k, v in d.items():
            parser.add_argument("--{}".format(k), default=v['val'], type=type(v['val']), help=v['desc'])
        args, unknown = parser.parse_known_args()
        return args
    
    def generate_all_path(self, args):
        # results path
        mid_dir = creat_check_path(args.output_path, 'processed', args.dataname)
        self.res_mid_dir = mid_dir
        if 'bert' in args.backbone:
            name = "processed_{}_{}_{}.pkl".format(args.seed, args.known_cls_ratio, args.labeled_ratio)
        elif 'cnn' in args.backbone:
            name = "processed_{}_{}_{}_{}.pkl".format(args.backbone, args.seed, args.known_cls_ratio, args.labeled_ratio)
        self.res_path = os.path.join(mid_dir, name)