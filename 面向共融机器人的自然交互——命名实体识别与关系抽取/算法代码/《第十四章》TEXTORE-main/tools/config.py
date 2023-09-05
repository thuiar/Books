import argparse
import os
from .utils import load_yaml, creat_check_path, get_logger

class Param:
    def __init__(self):
        self.base_param = 'tools/base_params.yaml'
        args = self.get_base_param()
        # get logger
        mid_dir = creat_check_path(args.result_path, args.task_type, args.method, "logs")
        self.logger = get_logger(mid_dir, args.task_type)
        if args.is_pipe == 0:
            args = self.get_method_param(args)

        for k, v in args.__dict__.items():
            setattr(self, k, v)
        if args.is_pipe == 0:
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
    def get_method_param(self, args):
        # get method settings
        self.logger.info("Get method config:")
        config_path = os.path.join(args.task_type, 'configs', '{}.yaml'.format(args.method))
        d = load_yaml(config_path)
        parser = argparse.ArgumentParser(allow_abbrev=False)
        for k, v in d.items():
            parser.add_argument("--{}".format(k), default=v['val'], type=type(v['val']), help=v['desc'])
        all_args, unknown = parser.parse_known_args()

        args_dict = all_args.__dict__
        input_args_dict = args.__dict__
        for key in args_dict:
            input_args_dict[key] = args_dict[key]
        return args
    def generate_all_path(self, args):
        # 1. model parameters
        temp_name = [args.dataname, args.this_name, args.seed, args.lr, args.known_cls_ratio, args.labeled_ratio]
        weight_name = "weights_{}.pth".format(
                    "-".join([str(x) for x in temp_name])
                )
        mid_dir = creat_check_path(args.save_path, args.task_type, args.method)
        output_model_file = os.path.join(mid_dir, weight_name)
        self.output_model_file = output_model_file
        # 2. pretrain path
        mid_dir = creat_check_path(args.save_path, args.task_type, args.method, "pretrain")
        pretrain_model_path = os.path.join(mid_dir, weight_name)
        self.pretrain_model_file = pretrain_model_path
        # 3. results path
        mid_dir = creat_check_path(args.result_path, args.task_type, args.method)
        self.res_mid_dir = mid_dir
        name = "{}_{}_{}_{}_{}.pkl".format(args.dataname, args.seed, args.this_name, args.known_cls_ratio, args.labeled_ratio)
        self.res_path = os.path.join(mid_dir, name)