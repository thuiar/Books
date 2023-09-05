from tools.utils import *

from .model import DTGNS
       
class Manager(BaseManager):
    def __init__(self, args, data):
        super(Manager, self).__init__(args)
        self.model = DTGNS(args, data.num_labels)
        if args.freeze_bert_parameters:
            self.freeze_bert_parameters(self.model)
       
        self.model.to(self.device)
        self.num_train_optimization_steps = int(len(data.train_feat) / args.train_batch_size) * args.num_train_epochs
        self.optimizer = self.get_optimizer(args, self.model)
        self.scheduler = self.get_scheduler(
            args, 
            self.optimizer, 
            args.warmup_proportion*self.num_train_optimization_steps,
            self.num_train_optimization_steps
        )

        self.best_eval_score = 0

        self.test_results = None
        self.predictions = None
        self.true_labels = None

    def get_pred_label(self, data, dataloader, need_feat=False):
        self.model.eval()
        total_labels = torch.empty(0, dtype=torch.long)
        total_preds = torch.empty(0, dtype=torch.long)
        hiddens = []
        for batch in tqdm(dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, pos, label_ids = batch
            with torch.set_grad_enabled(False):
                features = self.model(input_ids, input_mask, pos)
                hiddens.append(features.cpu())
                preds = self.model.classifier.predict(features, data.unseen_token_id)
                total_labels = torch.cat((total_labels, label_ids.cpu()))
                total_preds = torch.cat((total_preds, preds.cpu()))
        y_pred = total_preds.numpy()
        y_true = total_labels.numpy()
        hiddens = torch.cat(hiddens, dim=0)
        if need_feat:
            return y_true, y_pred, hiddens

        return y_true, y_pred

    def eval(self, args, data, mode="eval", show=False, is_test=False):
        if is_test:
            mode = 'test'
        if mode == 'eval':
            y_true, y_pred= self.get_pred_label(data, data.eval_dataloader)
            eval_acc = round(accuracy_score(y_true, y_pred) * 100, 2)

            return eval_acc

        elif mode == 'test':

            y_true, y_pred, hiddens = self.get_pred_label(data, data.test_dataloader, need_feat=True)
            cm = confusion_matrix(y_true, y_pred)
            results = F_measure(cm)
            acc = round(accuracy_score(y_true, y_pred) * 100, 2)
            results['Acc'] = acc
            self.test_results = results
            self.saver.save_results(args, results)
            reduce_dim = get_dimension_reduction(hiddens.numpy(), args.seed)
            self.saver.features = hiddens.numpy()
            self.saver.reduce_feat = reduce_dim
            self.saver.results = results
            self.saver.pred = self.predictions
            self.saver.labels = self.true_labels
            self.saver.known_label_list = data.known_label_list
            self.saver.all_label_list = data.all_label_list
            self.saver.save_output_results()

    def train(self, args, data):

        wait = 0
        self.best_eval_score = 0
        best_loss = 1e10
        # optimizer = self.get_optimizer(args, self.model)
        for epoch in range(int(args.num_train_epochs)):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            self.logger.info("Epoch: {} Iteration".format(epoch + 1))
            td = tqdm(data.train_dataloader, desc="Epoch: {} Iteration".format(epoch + 1))
            for step, batch in enumerate(td):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, pos, label_ids = batch
                with torch.set_grad_enabled(True):
                    loss = self.model(input_ids, input_mask, pos, label_ids, train=True)

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1)  # 79 没加这个s
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    tr_loss += loss.item()
                    
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
                    td.set_postfix(loss = tr_loss / nb_tr_steps)
                # break

            loss = tr_loss / nb_tr_steps
            self.logger.info('train_loss: {}'.format(loss))
            self.saver.append_train_loss(loss)
            eval_score = self.eval(args, data, mode='eval')
            self.saver.append_val_acc(eval_score)
            self.logger.info('eval_acc: {}'.format(eval_score))
            
            if eval_score > self.best_eval_score and epoch > 3:
                self.best_eval_score = eval_score
                wait = 0
                self.save_model(args)
            elif self.best_eval_score > 0.0 and eval_score > 0:
                wait += 1
                if wait >= args.wait_patient: # ran m1
                    break
            self.logger.info('cur_wait: {}'.format(wait))
        self.saver.save_middle()


    def class_count(self, labels):
        class_data_num = []
        for l in np.unique(labels):
            num = len(labels[labels == l])
            class_data_num.append(num)
        return class_data_num
    
    def restore_model(self, args):
        ckpt = restore_model(args)
        self.model.load_state_dict(ckpt['model'])

    def save_model(self, args):
        state_dict = {'model':self.model.state_dict()}
        save_model(args, state_dict)
