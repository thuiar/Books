from tools.utils import *
from .model import OpenMax
from .openmax_utils import recalibrate_scores, weibull_tailfitting, compute_distance
class Manager(BaseManager):
    
    def __init__(self, args, data):
        super(Manager, self).__init__(args)
        self.model = OpenMax(args, data.num_labels)
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
        self.feat_dim = self.model.encoder.out_dim
        self.best_eval_score = 0
        self.openmax_pred = None
        self.softmax_pred = None
        self.test_results = None
        self.predictions = None
        self.true_labels = None
        self.weibull_model = None
        self.feats = None

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
        self.save_model(args)
        self.saver.save_middle()
    def train(self, args, data):     
        
        self.logger.info('Training Start...')
        best_model = None
        wait = 0
        best_eval_score = 0
        train_results = []

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

            y_true, y_pred = self.get_outputs(args, data, mode = 'eval') 
            eval_score = round(accuracy_score(y_true, y_pred) * 100, 2)
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

        self.logger.info('Training finished...')
        self.saver.save_middle()

    def get_outputs(self, args, data, mode = 'eval', get_feats = False, \
        compute_centroids=False, get_probs = False):
        
        if mode == 'eval':
            dataloader = data.eval_dataloader
        elif mode == 'test':
            dataloader = data.test_dataloader
        elif mode == 'train':
            dataloader = data.train_dataloader

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long)
        total_logits = torch.empty((0, data.num_labels))
        total_features = torch.empty((0,self.feat_dim))
        
        centroids = torch.zeros(data.num_labels, data.num_labels).to(self.device)

        for batch in tqdm(dataloader, desc="Iteration"):
            input_ids, input_mask, pos, label_ids = self.to_cuda(*batch)
            with torch.set_grad_enabled(False):

                pooled_output, logits = self.model(input_ids, input_mask, pos)

                total_labels = torch.cat((total_labels, label_ids.cpu()))
                total_logits = torch.cat((total_logits, logits.cpu()))
                total_features = torch.cat((total_features, pooled_output.cpu()))

                if compute_centroids:
                    for i in range(len(label_ids)):
                        centroids[label_ids[i]] += logits[i]  
        self.feats = total_features.numpy()
        if get_feats:

            feats = total_features.numpy()
            return feats 

        else:

            total_probs = F.softmax(total_logits.detach(), dim=1)
            total_maxprobs, total_preds = total_probs.max(dim = 1)
            y_probs = total_probs.numpy()

            if get_probs:
                y_prob = total_maxprobs.numpy()
                return y_prob

            y_pred = total_preds.numpy()
            y_true = total_labels.numpy()

            y_logit = total_logits.numpy()

            if compute_centroids:
                
                centroids /= torch.tensor(self.class_count(y_true)).float().unsqueeze(1).to(self.device)
                centroids = centroids.detach().cpu().numpy()

                mean_vecs, dis_sorted = self.cal_vec_dis(args, data, centroids, y_logit, y_true)
                weibull_model = weibull_tailfitting(mean_vecs, dis_sorted, data.num_labels, tailsize = args.weibull_tail_size)
                
                return weibull_model

            else:

                if self.weibull_model is not None:

                    y_pred = self.classify_openmax(args, data, len(y_true), y_probs, y_logit)

                
            return y_true, y_pred


    def eval(self, args, data, is_test = True):
        self.weibull_model = self.get_outputs(args, data, mode = 'train', compute_centroids=True)
        y_true, y_pred = self.get_outputs(args, data, mode = 'test')
        y_prob = self.get_outputs(args, data, mode = 'test', get_probs = True)

        cm = confusion_matrix(y_true,y_pred)
        test_results = F_measure(cm)

        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        test_results['Acc'] = acc

        hiddens = self.feats
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
                'y_prob':y_prob,
                'openmax_pred':self.openmax_pred,
                'softmax_pred':self.softmax_pred
            }
        )
        self.save_final_results(args)
        return results
    
    def classify_openmax(self, args, data, num_samples, y_prob, y_logit):
            
        y_preds = []
        openmax_preds = []
        softmax_preds = []

        for i in range(num_samples):

            textarr = {}
            textarr['scores'] = y_prob[i]
            textarr['fc8'] = y_logit[i]
            openmax, softmax = recalibrate_scores(self.weibull_model, data.num_labels, textarr, \
                alpharank=min(args.alpharank, data.num_labels))

            openmax = np.array(openmax)
            pred = np.argmax(openmax)
            max_prob = max(openmax)
            
            openmax_preds.append(pred)
            softmax_preds.append(np.argmax(np.array(softmax)))

            if max_prob < args.threshold:
                pred = data.unseen_token_id

            y_preds.append(pred)    

        self.openmax_pred = openmax_preds
        self.softmax_pred = softmax_preds

        return y_preds

    def cal_vec_dis(self, args, data, centroids, y_logit, y_true):

        mean_vectors = [x for x in centroids]

        dis_all = []
        for i in range(data.num_labels):
            arr = y_logit[y_true == i]
            dis_all.append(self.get_distances(args, arr, mean_vectors[i]))

        dis_sorted = [sorted(x) for x in dis_all]

        return mean_vectors, dis_sorted    
    
    def get_distances(self, args, arr, mav):

        pre = []
        for i in arr:
            pre.append(compute_distance(i, mav, args.distance_type))

        return pre

    def class_count(self, labels):

        class_data_num = []
        for l in np.unique(labels):
            num = len(labels[labels == l])
            class_data_num.append(num)

        return class_data_num

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
