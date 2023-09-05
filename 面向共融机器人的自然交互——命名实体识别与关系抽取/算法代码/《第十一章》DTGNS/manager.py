from utils import *
from torch import nn, optim
class ModelManager:
    
    def __init__(self, args, model, data):
        self.model = model
        if args.freeze_bert_parameters:
            self.freeze_bert_parameters(self.model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.num_train_optimization_steps = int(len(data.train_examples) / args.train_batch_size) * args.num_train_epochs
        self.optimizer = self.get_optimizer(args)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=args.warmup_proportion*self.num_train_optimization_steps, 
            num_training_steps=self.num_train_optimization_steps)

        self.best_eval_score = 0

        self.test_results = None
        self.predictions = None
        self.true_labels = None

    def get_pred_label(self, data, dataloader):
        self.model.eval()
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_preds = torch.empty(0, dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, h, t, label_ids = batch
            with torch.set_grad_enabled(False):
                features = self.model(input_ids, input_mask, h, t)
                preds = self.model.predict(features, data.unseen_token_id)
                total_labels = torch.cat((total_labels, label_ids))
                total_preds = torch.cat((total_preds, preds))
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        return y_true, y_pred

    def evaluation(self, args, data, mode="eval", show=False):

        if mode == 'eval':
            y_true, y_pred = self.get_pred_label(data, data.eval_dataloader)
            eval_acc = round(accuracy_score(y_true, y_pred) * 100, 2)

            return eval_acc

        elif mode == 'test':

            y_true, y_pred = self.get_pred_label(data, data.test_dataloader)

            self.predictions = list([data.label_list[idx] for idx in y_pred])
            self.true_labels = list([data.label_list[idx] for idx in y_true])

            cm = confusion_matrix(y_true, y_pred)
            results = F_measure(cm)
            acc = round(accuracy_score(y_true, y_pred) * 100, 2)
            results['Acc'] = acc
            self.test_results = results

            if show:
                print('test_results', results)

    def get_optimizer(self, args):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                         lr = args.lr,
                         correct_bias=False)
        return optimizer

    def train(self, args, data):
        
        wait = 0
        self.best_eval_score = 0
        for epoch in range(int(args.num_train_epochs)):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(data.train_dataloader, desc="Epoch: {} Iteration".format(epoch + 1))):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, h, t, label_ids = batch
                with torch.set_grad_enabled(True):
                    loss = self.model(input_ids, input_mask, h, t, label_ids, train=True)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    tr_loss += loss.item()
                    
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

            loss = tr_loss / nb_tr_steps
            print('train_loss',loss)
            eval_score = self.evaluation(args, data, mode='eval')
            print('eval_acc', eval_score)
            
            
            if eval_score > self.best_eval_score:
                self.best_eval_score = eval_score
                wait = 0
                self.save_model(args)
            elif self.best_eval_score > 0 and eval_score > 0:
                wait += 1
                if wait >= args.wait_patient:
                    break
            print('cur_wait', wait)

    def freeze_bert_parameters(self, model):
        for name, param in model.bert.named_parameters():
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    def save_results(self, args):

        method_dir = os.path.join('methods', args.method)
        args.save_results_path = os.path.join(method_dir, args.save_results_path)
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        var = [args.dataset, args.method, args.known_cls_ratio, args.labeled_ratio, args.seed]
        names = ['dataset', 'method', 'known_cls_ratio', 'labeled_ratio', 'seed']
        vars_dict = {k: v for k, v in zip(names, var)}
        results = dict(self.test_results, **vars_dict)
        keys = list(results.keys())
        values = list(results.values())

        result_file = 'results_{}.csv'.format(args.this_name)
        results_path = os.path.join(args.save_results_path, result_file)

        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori, columns=keys)
            df1.to_csv(results_path, index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results, index=[1])
            df1 = df1.append(new, ignore_index=True)
            df1.to_csv(results_path, index=False)
        data_diagram = pd.read_csv(results_path)

        print('test_results', data_diagram)

    def save_model(self, args, name=None):
        print("save best acc ckpt!")
        weight_name = "weight_{}_{}_{}_{}.pth".format(args.seed, args.known_cls_ratio, args.dataset, args.this_name)
        mid_dir = args.save_path
        for temp_dir in [args.type, args.method]:
            mid_dir = os.path.join(mid_dir, temp_dir)
            if not os.path.exists(mid_dir):
                os.makedirs(mid_dir)
        model_file = os.path.join(mid_dir, weight_name)
        torch.save({'model':self.model.state_dict()}, model_file)
    def restore_model(self, args, name=None):
        weight_name = "weight_{}_{}_{}_{}.pth".format(args.seed, args.known_cls_ratio, args.dataset, args.this_name)
        mid_dir = os.path.join(args.save_path, args.type, args.method)
        output_model_file = os.path.join(mid_dir, weight_name)
        ckpt = torch.load(output_model_file)
        self.model.load_state_dict(ckpt['model'])
