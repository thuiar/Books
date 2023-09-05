from tools.utils import *
from .boundary import BoundaryLoss, euclidean_metric
from .model import ADB

class Manager(BaseManager):
    
    def __init__(self, args, data):
        super(Manager, self).__init__(args)
        self.model = ADB(args, data.num_labels)
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
        
        self.data = data
        self.train_dataloader = data.train_dataloader
        self.eval_dataloader = data.eval_dataloader
        self.test_dataloader = data.test_dataloader

        self.loss_fct = nn.CrossEntropyLoss()
        self.feat_dim = self.model.encoder.out_dim
        self.mid_dir = creat_check_path(args.save_path, args.task_type, args.method)
        if args.train_model:
            
            self.delta = None
            self.delta_points = []
            self.centroids = None
            self.train_results = []

        else:

            self.model = self.restore_model(args)
            self.delta = np.load(os.path.join(self.mid_dir, 'deltas.npy'))
            self.delta = torch.from_numpy(self.delta).to(self.device)
            self.centroids = np.load(os.path.join(self.mid_dir, 'centroids.npy'))
            self.centroids = torch.from_numpy(self.centroids).to(self.device)

    def pre_train(self, args, data):
        
        self.logger.info('Pre-training Start...')
        wait = 0
        best_model = None
        best_eval_score = 0

        for epoch in range(int(args.num_train_epochs)):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(data.train_dataloader, desc="Pre-Train: Epoch: {} Iteration".format(epoch + 1))):
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
            
            y_true, y_pred, _ = self.get_outputs(args, data, mode = 'eval', pre_train=True)
            eval_score = round(accuracy_score(y_true, y_pred) * 100, 2)

            eval_results = {
                'train_loss': loss,
                'eval_acc': eval_score,
                'best_acc':best_eval_score,
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

        self.logger.info('Pre-training finished...')


    def train(self, args, data):
        # if not os.path.isfile(args.output_model_file):
        self.pre_train(args, data)   
        self.restore_model(args)
        criterion_boundary = BoundaryLoss(num_labels = data.num_labels, feat_dim = self.feat_dim).to(self.device)
        
        self.delta = F.softplus(criterion_boundary.delta)
        optimizer = torch.optim.Adam(criterion_boundary.parameters(), lr = args.lr_boundary)
        self.centroids = self.centroids_cal(args, data)

        best_eval_score, best_delta, best_centroids = 0, None, None
        wait = 0

        train_results = []

        for epoch in range(int(args.num_train_epochs)):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(data.train_dataloader, desc="Epoch: {} Iteration".format(epoch + 1))):
                input_ids, input_mask, pos, label_ids = self.to_cuda(*batch)
                with torch.set_grad_enabled(True):
                    features = self.model(input_ids, input_mask, pos, feature_ext=True)
                    loss, self.delta = criterion_boundary(features, self.centroids, label_ids)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    tr_loss += loss.item()
                    
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

            self.delta_points.append(self.delta)

            loss = tr_loss / nb_tr_steps
            
            y_true, y_pred, _ = self.get_outputs(args, data, mode = 'eval')
            eval_score = round(f1_score(y_true, y_pred, average='macro') * 100, 2)

            eval_results = {
                'train_loss': loss,
                'eval_score': eval_score,
                'best_eval_score': best_eval_score,
            }
            train_results.append(eval_results)

            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            if eval_score > best_eval_score:

                wait = 0
                best_delta = self.delta 
                best_centroids = self.centroids
                best_eval_score = eval_score

            else:
                if best_eval_score > 0:
                    wait += 1
                    if wait >= args.wait_patient:
                        break

        self.delta = best_delta
        self.centroids = best_centroids
        self.train_results = train_results

        mid_dir = self.mid_dir
        np.save(os.path.join(mid_dir, 'centroids.npy'), self.centroids.detach().cpu().numpy())
        np.save(os.path.join(mid_dir, 'deltas.npy'), self.delta.detach().cpu().numpy())
            

    def get_outputs(self, args, data, mode = 'eval', get_feats = False, pre_train= False, delta = None):
        
        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long)
        total_preds = torch.empty(0,dtype=torch.long)
        
        total_features = torch.empty((0,self.feat_dim))
        total_logits = torch.empty((0, data.num_labels))
        
        for batch in tqdm(dataloader, desc="Iteration"):

            input_ids, input_mask, pos, label_ids = self.to_cuda(*batch)
            with torch.set_grad_enabled(False):

                pooled_output, logits = self.model(input_ids, input_mask, pos)
                
                if not pre_train:
                    preds = self.open_classify(data, pooled_output)
                    total_preds = torch.cat((total_preds, preds.cpu()))

                total_labels = torch.cat((total_labels,label_ids.cpu()))
                total_features = torch.cat((total_features, pooled_output.cpu()))
                total_logits = torch.cat((total_logits, logits.cpu()))

        if get_feats: 
            feats = total_features.numpy()
            return feats

        else:
    
            if pre_train:
                total_probs = F.softmax(total_logits.detach(), dim=1)
                total_maxprobs, total_preds = total_probs.max(dim = 1)

            y_pred = total_preds.numpy()
            y_true = total_labels.numpy()
            feats = total_features.numpy()
            
            return y_true, y_pred, feats


    def open_classify(self, data, features):

        logits = euclidean_metric(features, self.centroids)
        probs, preds = F.softmax(logits.detach(), dim = 1).max(dim = 1)
        euc_dis = torch.norm(features - self.centroids[preds], 2, 1).view(-1)
        preds[euc_dis >= self.delta[preds]] = data.unseen_token_id

        return preds
    
    def eval(self, args, data, is_test=True):
        
        # hiddens = self.get_outputs(args, data, mode = 'test', get_feats = True)
        y_true, y_pred, hiddens = self.get_outputs(args, data, mode = 'test')
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)

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
        self.saver.save_output_results()
        self.save_final_results(args)
        return results

    def class_count(self, labels):
        class_data_num = []
        for l in np.unique(labels):
            num = len(labels[labels == l])
            class_data_num.append(num)
        return class_data_num

    def centroids_cal(self, args, data):
        
        self.model.eval()
        centroids = torch.zeros(data.num_labels, self.feat_dim)
        total_labels = torch.empty(0, dtype=torch.long)

        with torch.set_grad_enabled(False):

            for batch in self.train_dataloader:

                input_ids, input_mask, pos, label_ids = self.to_cuda(*batch)

                features = self.model(input_ids, input_mask, pos, feature_ext=True)
                total_labels = torch.cat((total_labels, label_ids.cpu()))
                for i in range(len(label_ids)):
                    label = label_ids[i]
                    centroids[label] += features[i].cpu()
                
        total_labels = total_labels.numpy()
        centroids /= torch.tensor(self.class_count(total_labels)).float().unsqueeze(1)
        centroids = centroids.to(self.device)
        
        return centroids
    def restore_model(self, args):
        ckpt = restore_model(args)
        self.model.load_state_dict(ckpt["ckpt"])
    
    def save_final_results(self, args):
        #save centroids, delta_points
        var = [args.method, args.known_cls_ratio, args.labeled_ratio]
        names = ['method', 'known_cls_ratio', 'labeled_ratio']
        vars_dict = {k:v for k,v in zip(names, var) }
        results = dict(self.test_results,**vars_dict)
        self.saver.save_results(args, self.test_results)

    def save_model(self, args):
        state_dict = {"ckpt": self.model.state_dict()}
        save_model(args, state_dict)
    
