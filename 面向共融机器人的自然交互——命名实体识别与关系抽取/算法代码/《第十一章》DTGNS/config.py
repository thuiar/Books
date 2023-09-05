import argparse

class Param:

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser = self.common_param(parser)
        args = parser.parse_args() 

        backbones = {'bert':self.bert} 
        parser = backbones[args.backbone](parser)

        self.parser = parser
    def common_param(self, parser):
        parser.add_argument("--dataset", default='semeval', type=str, 
                            help="The name of the dataset to train selected")
        
        parser.add_argument("--known_cls_ratio", default=0.75, type=float, help="The number of known classes")
        
        parser.add_argument("--labeled_ratio", default=1.0, type=float, help="The ratio of labeled samples in the training set")
        
        parser.add_argument("--method", type=str, default='DTGNS', help="which method to use")

        parser.add_argument("--backbone", type=str, default='bert', help="which model to use")

        parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")

        parser.add_argument('--type', type=str, default='open_relation_classification', help="Type for methods")

        parser.add_argument("--num_train_epochs", default=100, type=float,
                            help="Total number of training epochs to perform.") 

        parser.add_argument("--this_name", default="", type=str,
                            help="Save weight name.")

        parser.add_argument("--mlp_hidden", default=100, type=int,
                              help="Hidden size of MLP layer")
        return parser

    def bert(self, parser):
        ##############Your Location for Pretrained Bert Model#####################
        parser.add_argument("--bert_model", default="bert-base-uncased/", type=str, help="The path for the pre-trained bert model.")

        parser.add_argument("--data_dir", default='data/', type=str,
                            help="The input data dir. Should contain the .csv files (or other data files) for the task.")
        
        parser.add_argument("--save_results_path", type=str, default='outputs', help="the path to save results")

        parser.add_argument("--save_path", default='weight_cache/', type=str, 
                            help="The output directory where the model predictions and checkpoints will be written.") 

        parser.add_argument("--feat_dim", default=768, type=int, help="The feature dimension.")

        parser.add_argument("--warmup_proportion", default=0.1, type=float)

        parser.add_argument("--freeze_bert_parameters", action="store_true", default='freeze', help="Freeze parameters of BERT")

        parser.add_argument("--save_model", action="store_true", help="save trained-model")

        parser.add_argument("--save_results", action="store_true", help="save test results")
        
        parser.add_argument("--gpu_id", type=str, default='0', help="Select the GPU id")

        parser.add_argument("--lr", default=2e-5, type=float,
                            help="The learning rate of BERT.")    
        
        parser.add_argument("--train_batch_size", default=64, type=int,
                            help="Batch size for training.")
        
        parser.add_argument("--eval_batch_size", default=64, type=int,
                            help="Batch size for evaluation.")
        
        parser.add_argument("--wait_patient", default=15, type=int,
                            help="Patient steps for Early Stop.")    
        
        return parser