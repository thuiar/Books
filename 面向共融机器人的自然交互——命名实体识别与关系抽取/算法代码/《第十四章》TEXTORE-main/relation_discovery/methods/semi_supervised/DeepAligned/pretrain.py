from tools.utils import *
from .model import BERT
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

class PretrainManager(BaseManager):
    """
    from https://github.com/thuiar/TEXTOIR
    """
    def __init__(self, args, data=None):
        super(PretrainManager, self).__init__(args)
        self.model = BERT(args).to(self.device)
        self.optimizer = self.get_optimizer(args, self.model)
        num_train_examples = len(data.train_feat)
        num_train_optimization_steps = int(num_train_examples / args.train_batch_size) * args.num_train_epochs
        self.scheduler = self.get_scheduler(
            args, 
            self.optimizer, 
            warmup_step=args.warmup_proportion * num_train_optimization_steps,
            train_iter=num_train_optimization_steps
        )
        if args.freeze_bert_parameters:
            self.freeze_bert_parameters(self.model)
        self.train_dataloader = data.sup_train_dataloader
        self.eval_dataloader = data.eval_dataloader
        self.test_dataloader = data.test_dataloader 
        self.loss_fct = nn.CrossEntropyLoss()


    def train(self, args, data):

        wait = 0
        best_model = None
        best_eval_score = 0

        for epoch in trange(int(args.num_pretrain_epochs), desc="Epoch"):
            
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):
                input_ids, batch_mask, batch_pos, label_ids= self.to_cuda(*batch)
                # batch = tuple(t.to(self.device) for t in batch)
                # input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):

                    loss = self.model(input_ids, batch_mask, batch_pos, label_ids, mode = "train", loss_fct = self.loss_fct)
                    
                    loss.backward()
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
            
            loss = tr_loss / nb_tr_steps
            
            y_true, y_pred = self.get_outputs(args, mode = 'eval')
            eval_score = round(accuracy_score(y_true, y_pred) * 100, 2)

            eval_results = {
                'train_loss': loss,
                'eval_score': eval_score,
                'best_score':best_eval_score,
            }
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            if eval_score > best_eval_score:
                self.save_model(args)
                wait = 0
                best_eval_score = eval_score

            elif eval_score > 0:

                wait += 1
                if wait >= args.wait_patient:
                    break
                
    def get_outputs(self, args, mode = 'eval', get_feats = False):
        
        if mode == 'eval':
            dataloader = self.eval_dataloader

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)
        
        for batch in tqdm(dataloader, desc="Iteration"):
            input_ids, batch_mask, batch_pos, label_ids= self.to_cuda(*batch)
            # batch = tuple(t.to(self.device) for t in batch)
            # input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                pooled_output, logits = self.model(input_ids, batch_mask, batch_pos)
                
                total_labels = torch.cat((total_labels,label_ids))
                total_features = torch.cat((total_features, pooled_output))
                total_logits = torch.cat((total_logits, logits))

        if get_feats:  
            feats = total_features.cpu().numpy()
            return feats 

        else:
            total_probs = F.softmax(total_logits.detach(), dim=1)
            total_maxprobs, total_preds = total_probs.max(dim = 1)

            y_pred = total_preds.cpu().numpy()
            y_true = total_labels.cpu().numpy()

            return y_true, y_pred

    def restore_model(self, args):
        ckpt = restore_model(args)
        self.model.load_state_dict(ckpt['state_dict'])
    def save_model(self, args):
        save_dict = {'state_dict': self.model.state_dict()}
        # save_pretrain_model()
        save_model(args, save_dict)
