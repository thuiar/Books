import argparse
import os
class Param:

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser = self.all_param(parser)
        all_args, unknown = parser.parse_known_args()
        self.args = all_args

    def all_param(self, parser):
        ##################################common parameters####################################
        parser.add_argument("--dataname", default='fewrel', type=str)
        
        parser.add_argument("--data_path", default='data/datasets', type=str)

        parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")

        parser.add_argument('--task_type', type=str, default='relation_discover', help="Type for methods")

        parser.add_argument("--results_path", type=str, default='relation_discover/results', help="the path to save results of methods")
        
        parser.add_argument("--save_path", default='relation_discover/weight_cache', type=str)

        parser.add_argument("--rel_nums", default=64, type=float)

        parser.add_argument("--this_name", default="discover", type=str)

        parser.add_argument("--optim", default="adamw", type=str)

        parser.add_argument("--max_length", default=100, type=int)

        parser.add_argument("--num_worker", default=0, type=int)

        parser.add_argument("--learning_rate", default=1e-5, type=float)

        parser.add_argument("--sche", default="linear_warmup", type=str)
        
        parser.add_argument("--wait_patient", default=5, type=int)
        
        parser.add_argument("--train_model", default=1, type=int)

        parser.add_argument("--load_ckpt", default="", type=str)

        parser.add_argument("--pretrain_name", default="with_fewrel", type=str)

        parser.add_argument("--bert_path", default='/home/jaczhao/bert/bert-base-uncased', type=str)

        ###############################   Loss ################################################
        parser.add_argument("--pos_margin", default=0.7, type=float)

        parser.add_argument("--neg_margin", default=1.4, type=float)

        parser.add_argument("--temp", default=50, type=float)

        ###############################   Model Training ################################################
        parser.add_argument("--z_dim", default=768, type=int)

        parser.add_argument("--n_clusters", default=16, type=int)

        parser.add_argument("--batch_num", default=1000)

        parser.add_argument("--val_step", default=500)

        parser.add_argument("--batch_size", default=64, type=int)

        parser.add_argument("--loop_nums", default=4, type=int)

        parser.add_argument("--pre_loop_nums", default=4, type=int)
        
        parser.add_argument("--class_num_ratio", default=0.1) 

        parser.add_argument("--warmup_ratio", default=0.1) 
        
        return parser
