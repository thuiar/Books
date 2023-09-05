from tools.utils import *
from .model import MSP
class Manager(BaseManager):
    
    def __init__(self, args, data):
        super(Manager, self).__init__(args)
        self.model = MSP(args, data.num_labels)
        if args.freeze_bert_parameters:
            self.freeze_bert_parameters(self.model)
        
        self.model.to(self.device)
        self.num_train_optimization_steps = int(len(data.train_feat) / args.train_batch_size) * args.num_train_epochs
        self.optimizer = self.get_optimizer(args, self.model)
        self.scheduler = self.get_scheduler(
            args, self.optimizer,
            args.warmup_proportion*self.num_train_optimization_steps,
            self.num_train_optimization_steps
        )
        self.loss_fct = nn.CrossEntropyLoss()
        
        self.best_eval_score = 0
        self.best_mu_stds = None
        self.test_results = None
        self.predictions = None
        self.true_labels = None


    def get_prob_label(self, data, dataloader):

        self.model.eval()
        total_labels = torch.empty(0,dtype=torch.long)
        total_logits = torch.empty((0, data.num_labels))
        hiddens = []
        for batch in tqdm(dataloader, desc="Iteration"):
            input_ids, input_mask, pos, label_ids = self.to_cuda(*batch)
            with torch.set_grad_enabled(False):
                hidden, logits = self.model(input_ids, input_mask, pos)
                hiddens.append(hidden.cpu())
                total_labels = torch.cat((total_labels,label_ids.cpu()))
                total_logits = torch.cat((total_logits, logits.cpu()))
        hiddens = torch.cat(hiddens, dim=0)
        
        total_probs = F.softmax(total_logits.detach(), dim=1)
        y_prob = total_probs
        y_true = total_labels.numpy()

        return y_true, y_prob, hiddens
    

    def eval(self, args, data, is_test = True):
        dataloader = data.test_dataloader if is_test else data.eval_dataloader
        y_true, y_prob, hiddens = self.get_prob_label(data, dataloader)
        max_probs, y_pred = y_prob.max(dim=1)
        y_pred = y_pred.numpy()
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        if is_test:
            hiddens = hiddens.numpy()
            max_probs = max_probs.numpy()
            y_pred[max_probs < args.threshold] = data.unseen_token_id
            self.predictions = list([data.label_list[idx] for idx in y_pred])
            self.true_labels = list([data.label_list[idx] for idx in y_true])
            cm = confusion_matrix(y_true,y_pred)
            results = F_measure(cm)
            results['Acc'] = acc
            self.test_results = results
            reduce_dim = get_dimension_reduction(hiddens, args.seed)
            self.saver.features = hiddens
            self.saver.reduce_feat = reduce_dim
            self.saver.results = results
            self.saver.pred = self.predictions
            self.saver.labels = self.true_labels
            self.saver.known_label_list = data.known_label_list
            self.saver.all_label_list = data.all_label_list
            self.saver.save_output_results(
                {
                    'y_prob':y_prob.numpy()
                }
            )
            self.save_final_results(args)
            return results
        else:
            return acc

    def train(self, args, data):     
        best_model = None
        wait = 0

        for epoch in range(int(args.num_train_epochs)):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(data.train_dataloader, desc="Epoch: {} Iteration".format(epoch + 1))):
                input_ids, input_mask, pos, label_ids = self.to_cuda(*batch)
                with torch.set_grad_enabled(True):
                    loss = self.model(input_ids, input_mask, pos, label_ids, mode='train', loss_fct=self.loss_fct)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    tr_loss += loss.item()
                    
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

            loss = tr_loss / nb_tr_steps
            self.logger.info('train_loss: {}'.format(loss))
            self.saver.append_train_loss(loss)
            
            eval_score = self.eval(args, data, is_test=False)          
            self.logger.info("eval acc: {}".format(eval_score))
            self.saver.append_val_acc(eval_score)

            if eval_score >= self.best_eval_score:
                self.best_eval_score = eval_score
                self.save_model(args)
                wait = 0
                
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break
            self.logger.info("wait: {}".format(wait))
        self.saver.save_middle()

    def restore_model(self, args):
        ckpt = restore_model(args)
        self.model.load_state_dict(ckpt["ckpt"])
        self.best_eval_score = ckpt['best_eval']
    
    def save_final_results(self, args):
        #save centroids, delta_points
        
        var = [args.method, args.known_cls_ratio, args.labeled_ratio]
        names = ['method', 'known_cls_ratio', 'labeled_ratio']
        vars_dict = {k:v for k,v in zip(names, var) }
        results = dict(self.test_results,**vars_dict)
        self.saver.save_results(args, self.test_results)

    def save_model(self, args):
        state_dict = {"ckpt": self.model.state_dict(), "best_eval": self.best_eval_score}
        save_model(args, state_dict)
