import argparse
import os

class Param:

    def __init__(self, input_args=None):
        parser = argparse.ArgumentParser()
        parser = self.all_param(parser)
        all_args, unknown = parser.parse_known_args()
        
        if input_args is not None:
            args_dict = all_args.__dict__
            input_args_dict = input_args.__dict__
            for key in input_args_dict:
                args_dict[key] = input_args_dict[key]
        self.args = all_args

    def all_param(self, parser):
        
        parser.add_argument("--detection_method", type=str, default='DOC', help="which detection method to use")

        parser.add_argument("--discover_method", type=str, default='MORE', help="which discover method to use")

        parser.add_argument("--detection_weight_path", type=str, default='')

        parser.add_argument("--discover_weight_path", type=str, default='')

        return parser