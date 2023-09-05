from tools.utils import *
from .model import BertForConstrainClustering
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

class Manager(BaseManager):
    """
    from https://github.com/thuiar/TEXTOIR
    """
    def __init__(self, args, data=None):
        super(Manager, self).__init__(args)
        self.model = BertForConstrainClustering(args).to(self.device)
        self.optimizer1 = self.get_optimizer(args, self.model)
        num_train_examples = len(data.semi_feat)
        num_train_optimization_steps = int(num_train_examples / args.train_batch_size) * args.num_train_epochs
        self.scheduler1 = self.get_scheduler(
            args, 
            self.optimizer1, 
            warmup_step=args.warmup_proportion * num_train_optimization_steps,
            train_iter=num_train_optimization_steps
        )
        
        if args.freeze_bert_parameters:
            self.freeze_bert_parameters(self.model)
        self.train_labeled_dataloader = data.sup_train_dataloader
        self.train_unlabeled_dataloader = data.train_dataloader
        self.train_dataloader = data.semi_train_dataloader
        self.eval_dataloader = data.eval_dataloader
        self.test_dataloader = data.test_dataloader 
    def initialize_centroids(self, args, data):
        
        self.logger.info("Initialize centroids...")

        feats = self.get_outputs(args, mode = 'train_unlabeled', get_feats = True)
        km = KMeans(n_clusters=data.num_labels, n_jobs=-1, random_state=args.seed)
        km.fit(feats)

        self.logger.info("Initialization finished...")

        self.model.cluster_layer.data = torch.tensor(km.cluster_centers_).to(self.device)

    def train(self, args, data): 

        self.logger.info('Pairwise-similarity Learning begin...')
        
        u = args.u
        l = args.l
        eta = 0

        eval_pred_last = np.zeros_like(data.eval_feat)
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):  
            
            # Fine-tuning with auxiliary distribution
            tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0
            self.model.train()

            for step, batch in enumerate(tqdm(self.train_labeled_dataloader, desc="Epoch: {} Iteration (labeled)".format(epoch+1))):
                input_ids, batch_mask, batch_pos, label_ids = self.to_cuda(*batch)
                # batch = tuple(t.to(self.device) for t in batch)
                # input_ids, input_mask, segment_ids, label_ids = batch
                loss = self.model(input_ids, batch_mask, batch_pos, label_ids, u_threshold = u, l_threshold = l, mode = 'train')
                loss.backward()
                
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                self.optimizer1.step()
                self.scheduler1.step()
                self.optimizer1.zero_grad() 

            train_labeled_loss = tr_loss / nb_tr_steps

            tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Epoch: {} Iteration (all train)".format(epoch+1))):
                input_ids, batch_mask, batch_pos, label_ids = self.to_cuda(*batch)

                # batch = tuple(t.to(self.device) for t in batch)
                # input_ids, input_mask, segment_ids, label_ids = batch

                loss = self.model(input_ids, batch_mask, batch_pos, label_ids, u_threshold = u, l_threshold = l, mode = 'train', semi = True)
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                self.optimizer1.step()
                self.scheduler1.step()
                self.optimizer1.zero_grad()
            
            train_loss = tr_loss / nb_tr_steps
            self.saver.append_train_loss(train_loss)

            eval_true, eval_pred = self.get_outputs(args, mode = 'eval')
            eval_score = clustering_score(eval_true, eval_pred)['NMI']
            self.saver.append_val_acc(eval_score)

            delta_label = np.sum(eval_pred != eval_pred_last).astype(np.float32) / eval_pred.shape[0]
            eval_pred_last = np.copy(eval_pred)

            train_results = {
                'u_threshold': round(u, 4),
                'l_threshold': round(l, 4),
                'train_labeled_loss': train_labeled_loss,
                'train_loss': train_loss,
                'delta_label': delta_label,
                'eval_score': eval_score
            }
            
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch))
            for key in sorted(train_results.keys()):
                self.logger.info("  %s = %s", key, str(train_results[key]))
            
            eta += 1.1 * 0.009
            u = 0.95 - eta
            l = 0.455 + eta * 0.1
            if u < l:
                break
        
        self.logger.info('Pairwise-similarity Learning finished...')

        self.refine(args, data)

    def refine(self, args, data):
        
        self.optimizer2 = self.get_optimizer(args, self.model)
        num_train_examples = len(data.unlabel_train_feat)
        num_train_optimization_steps = int(num_train_examples / args.train_batch_size) * args.num_refine_epochs
        self.scheduler2 = self.get_scheduler(
            args, 
            self.optimizer2, 
            warmup_step=args.warmup_proportion * num_train_optimization_steps,
            train_iter=num_train_optimization_steps
        )

        self.logger.info('Cluster refining begin...')
        self.initialize_centroids(args, data)
        wait = 0
        train_preds_last = None
        best_eval_score = 0

        for epoch in range(args.num_refine_epochs):
            
            #evaluation
            eval_true, eval_pred = self.get_outputs(args, mode = 'eval')
            eval_score = clustering_score(eval_true, eval_pred)['NMI']

            #early stop
            if eval_score > best_eval_score:
                wait = 0
                best_eval_score = eval_score
                self.save_model(args)

            else:
                wait += 1
                if wait > args.wait_patient:
                    break

            #converge
            train_pred_logits = self.get_outputs(args, mode = 'train', get_logits = True)
            p_target = target_distribution(train_pred_logits)
            train_preds = train_pred_logits.argmax(1)

            delta_label = np.sum(train_preds != train_preds_last).astype(np.float32) / train_preds.shape[0]
            train_preds_last = np.copy(train_preds)

            if epoch > 0 and delta_label < 0.001:
                self.logger.info('Break at epoch: %s and delta_label: %f.', str(epoch + 1), round(delta_label, 2))
                break
            
            # Fine-tuning with auxiliary distribution
            self.model.train()
            tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0

            for step, batch in enumerate(self.train_dataloader):
                input_ids, batch_mask, batch_pos, label_ids= self.to_cuda(*batch)
                # batch = tuple(t.to(self.device) for t in batch)
                # input_ids, input_mask, segment_ids, label_ids = batch
                feats, logits = self.model(input_ids, batch_mask, batch_pos, mode='finetune')
                kl_loss = F.kl_div(logits.log(), torch.Tensor(p_target[step * args.train_batch_size: (step + 1) * args.train_batch_size]).cuda())
                kl_loss.backward()

                tr_loss += kl_loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                self.optimizer2.step()
                self.scheduler2.step()
                self.optimizer2.zero_grad() 
            
            train_loss = tr_loss / nb_tr_steps
            self.saver.append_train_loss(train_loss)
            eval_results = {
                'kl_loss': round(train_loss, 4), 
                'delta_label': delta_label.round(4),
                'eval_score': round(eval_score, 2),
                'best_eval_score': round(best_eval_score, 2)
            }
            self.saver.append_val_acc(eval_score)
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))

        self.logger.info('Cluster refining finished...')
    
    def get_outputs(self, args,  mode = 'eval', get_feats = False, get_logits = False):
        
        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader
        elif mode == 'train_unlabeled':
            dataloader = self.train_unlabeled_dataloader
        elif mode == 'train':
            dataloader = self.train_dataloader

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long)
        total_preds = torch.empty(0,dtype=torch.long)

        total_features = torch.empty((0, args.num_labels))
        total_logits = torch.empty((0, args.num_labels))

        for batch in tqdm(dataloader, desc="Iteration"):
            batch_word, batch_mask, batch_pos, label_ids= self.to_cuda(*batch)

            with torch.set_grad_enabled(False):
                pooled_output, logits = self.model(batch_word, batch_mask, batch_pos)
    
                total_labels = torch.cat((total_labels, label_ids.cpu()))
                total_features = torch.cat((total_features, pooled_output.cpu()))
                total_logits = torch.cat((total_logits, logits.cpu()))
        if mode=='test':
            total_preds = total_logits.argmax(1)
            y_pred = total_preds.numpy()
            y_true = total_labels.numpy()
            feats = total_features.numpy()
            return y_true, y_pred, feats

        if get_feats:
            feats = total_features.numpy()
            return feats

        elif get_logits:
            logits = total_logits.numpy()
            return logits

        else:
            total_preds = total_logits.argmax(1)
            y_pred = total_preds.numpy()
            y_true = total_labels.numpy()

            return y_true, y_pred

    def eval(self, args, eval_data, is_test=True):

        y_true, y_pred, hiddens = self.get_outputs(args, mode = 'test')
        results = clustering_score(y_true, y_pred) 
        cm = confusion_matrix(y_true,y_pred) 
        
        self.logger.info
        self.logger.info("***** Test: Confusion Matrix *****")
        self.logger.info("%s", str(cm))
        self.logger.info("***** Test results *****")
        
        for key in sorted(results.keys()):
            self.logger.info("  %s = %s", key, str(results[key]))


        reduce_dim = get_dimension_reduction(hiddens, args.seed)
        self.saver.features = hiddens
        self.saver.reduce_feat = reduce_dim
        self.saver.results = results
        self.saver.pred = y_pred
        self.saver.labels = eval_data.test_labels #list([eval_data.label_list[idx] for idx in labels])
        # self.saver.samples = eval_data.test_texts
        self.saver.known_label_list = eval_data.known_label_list
        self.saver.all_label_list = eval_data.all_label_list
        self.saver.save_output_results(
            {
                'y_true': y_true
            }
        )
        results['B3'] = round(results['B3']['F1']*100, 2)
        self.saver.save_results(args, results)

        return results

    def restore_model(self, args):
        ckpt = restore_model(args)
        self.model.load_state_dict(ckpt['state_dict'])
    def save_model(self, args):
        save_dict = {'state_dict': self.model.state_dict()}
        save_model(args, save_dict)
