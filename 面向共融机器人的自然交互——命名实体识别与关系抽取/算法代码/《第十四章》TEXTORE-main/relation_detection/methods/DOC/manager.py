from scipy.sparse.dia import dia_matrix
from tools.utils import *
from .model import DOC
from scipy.stats import norm as dist_model
class Manager(BaseManager):
    
    def __init__(self, args, data):
        super(Manager, self).__init__(args)
        self.model = DOC(args, data.num_labels)
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
        self.best_eval_score = 0
        self.best_mu_stds = None
        self.test_results = None
        self.predictions = None
        self.true_labels = None
        self.thresholds = None

    def classify_doc(self, data, args, probs, mu_stds):

        thresholds = {}
        for col in range(data.num_labels):
            threshold = max(0.5, 1 - args.scale * mu_stds[col][1])
            label = data.known_label_list[col]
            thresholds[label] = threshold

        self.logger.info('DOC_thresholds:{}'.format(thresholds))
        self.thresholds = thresholds
        preds = []
        for p in probs:
            max_class = np.argmax(p)
            max_value = np.max(p)
            threshold = max(0.5, 1 - args.scale * mu_stds[max_class][1])
            if max_value > threshold:
                preds.append(max_class)
            else:
                preds.append(data.unseen_token_id)

        return np.array(preds)

    def get_prob_label(self, data, dataloader):

        self.model.eval()
        total_labels = torch.empty(0,dtype=torch.long)
        total_logits = torch.empty((0, data.num_labels))
        hiddens = []
        for batch in tqdm(dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, pos, label_ids = batch
            with torch.set_grad_enabled(False):
                hidden, logits = self.model(input_ids, input_mask, pos)
                hiddens.append(hidden.cpu())
                total_labels = torch.cat((total_labels,label_ids.cpu()))
                total_logits = torch.cat((total_logits, logits.cpu()))
        hiddens = torch.cat(hiddens, dim=0)
        
        total_probs = torch.sigmoid(total_logits.detach())
        y_prob = total_probs.numpy()
        y_true = total_labels.numpy()

        return y_true, y_prob, hiddens
    
    def get_mu_stds(self, args, data):

        dataloader = data.train_dataloader
        y_true, y_prob, hiddens = self.get_prob_label(data, dataloader)
        mu_stds = self.cal_mu_std(y_prob, y_true, data.num_labels)
    
        return mu_stds

    def eval(self, args, data, is_test = True, mu_stds = None):
        dataloader = data.test_dataloader if is_test else data.eval_dataloader
        y_true, y_prob, hiddens = self.get_prob_label(data, dataloader)
        if mu_stds is None:
            mu_stds = self.best_mu_stds
        y_pred = self.classify_doc(data, args, y_prob, mu_stds)
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        if is_test:
            self.predictions = list([data.label_list[idx] for idx in y_pred])
            self.true_labels = list([data.label_list[idx] for idx in y_true])
            cm = confusion_matrix(y_true,y_pred)
            results = F_measure(cm)
            results['Acc'] = acc
            self.test_results = results
            reduce_dim = get_dimension_reduction(hiddens.numpy(), args.seed)
            self.saver.features = hiddens.numpy()
            self.saver.reduce_feat = reduce_dim
            self.saver.results = results
            self.saver.pred = self.predictions
            self.saver.labels = self.true_labels
            self.saver.known_label_list = data.known_label_list
            self.saver.all_label_list = data.all_label_list
            self.saver.save_output_results(
                {
                    'thresholds': self.thresholds,
                    'y_prob':y_prob
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
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, pos, label_ids = batch
                with torch.set_grad_enabled(True):
                    loss_fct = nn.CrossEntropyLoss()
                    loss = self.model(input_ids, input_mask, pos,label_ids, mode='train', loss_fct=loss_fct)

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
            
            mu_stds = self.get_mu_stds(args, data)
            eval_score = self.eval(args, data, is_test=False, mu_stds=mu_stds)          
            self.logger.info("eval acc: {}".format(eval_score))
            self.saver.append_val_acc(eval_score)

            if eval_score >= self.best_eval_score:
                self.best_eval_score = eval_score
                self.best_mu_stds = mu_stds
                self.save_model(args)
                wait = 0
                
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break
            self.logger.info("wait: {}".format(wait))
        self.saver.save_middle()
    
    def fit(self, prob_pos_X):
        prob_pos = [p for p in prob_pos_X] + [2 - p for p in prob_pos_X]
        pos_mu, pos_std = dist_model.fit(prob_pos)
        return pos_mu, pos_std

    def cal_mu_std(self, probs, trues, num_labels):

        mu_stds = []
        for i in range(num_labels):
            pos_mu, pos_std = self.fit(probs[trues == i, i])
            mu_stds.append([pos_mu, pos_std])

        return mu_stds

    def restore_model(self, args):
        ckpt = restore_model(args)
        self.model.load_state_dict(ckpt["ckpt"])
        self.best_eval_score = ckpt['best_eval']
        self.best_mu_stds = ckpt['mu_std']
    
    def save_final_results(self, args):
        #save centroids, delta_points
        self.saver.save_results(args, self.test_results)

    def save_model(self, args):
        state_dict = {"ckpt": self.model.state_dict(), "mu_std":self.best_mu_stds, "best_eval": self.best_eval_score}
        save_model(args, state_dict)





  

    
    
