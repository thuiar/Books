from tools.utils import *
from .rsn_loader import RSNDataloader

class Manager(BaseManager):
    def __init__(self, args, data=None):
        super(Manager, self).__init__(args)
        if args.backbone in ['cnn']:
            from .model import CNNModel
            args.lr = args.learning_rate
            self.model = CNNModel(args, self.device).to(self.device)
            self.optimizer = self.get_optimizer(args, self.model)
            self.scheduler = None
        elif args.backbone in ['bert']:
            from .model_bert import BERTModel
            # args.lr = args.learning_rate
            self.model = BERTModel(args, self.device).to(self.device)
            if args.freeze_bert_parameters:
                self.freeze_bert_parameters(self.model)
            self.optimizer = self.get_optimizer(args, self.model)
            train_iters = args.batch_num * args.epoch_num
            self.scheduler = self.get_scheduler(args, self.optimizer, warmup_step=train_iters * args.warmup_proportion, train_iter=train_iters)
        self.lambda_s = args.lambda_s
        self.p_cond = args.p_cond

    def train(self, args, data):
        best_score = 0
        labeled_train_set = data._sup_train_dataloader.dataset.data
        unlabeled_train_set = data.train_dataloader.dataset.data
        labeled_train_loader = RSNDataloader(labeled_train_set, args)
        unlabeled_train_loader = RSNDataloader(unlabeled_train_set, args)

        # preparing testing samples
        val_left_input, val_right_input, val_label = \
        labeled_train_loader.next_batch(args.val_size, args.same_ratio)
        

        batch_num_list = [args.batch_num] * args.epoch_num

        for epoch in range(args.epoch_num): # loop 4
            self.logger.info('------epoch {}------'.format(epoch))
            self.logger.info('max batch num to train is {}'.format(batch_num_list[epoch]))
            self.model.train()
            total_loss = 0.0
            t = tqdm(range(1, batch_num_list[epoch]+1), desc="Epoch: {} Training".format(epoch+1))
            for i in t:
                self.model.train()
                train_loss = self.train_semi(labeled_train_loader, unlabeled_train_loader, args.batch_size, args.same_ratio)
                total_loss += train_loss
                t.set_postfix(train_loss= total_loss / i)
                # eval
                self.saver.append_train_loss(train_loss)
                self.logger.info("train loss: {}".format(train_loss))
                if i % 1000 == 0:
                    self.model.eval()
                    acc, tp, fp, fn, tn = 0.0,0.0,0.0,0.0,0.0
                    total = 0
                    val_len = len(val_left_input[0])
                    for i in tqdm(range(args.val_size)):
                        if i*2>=val_len:
                            break
                        total += 1
                        temp_val_left_input = (val_left_input[0][i*2:(i+1)*2], val_left_input[1][i*2:(i+1)*2], val_left_input[2][i*2:(i+1)*2])
                        temp_val_right_input = (val_right_input[0][i*2:(i+1)*2], val_right_input[1][i*2:(i+1)*2], val_right_input[2][i*2:(i+1)*2])
                        # if temp_val_left_input[0].size(0) < 1:
                        #     break
                        temp_val_left_input = self.to_cuda(*temp_val_left_input)
                        temp_val_right_input = self.to_cuda(*temp_val_right_input)
                        temp_val_label = val_label[i*2:(i+1)*2].to(self.device)
                        val_trainset_info = self.model.validation(temp_val_left_input, temp_val_right_input, temp_val_label)
                        acc += val_trainset_info[0]
                        tp += val_trainset_info[1]
                        fp += val_trainset_info[2]
                        fn += val_trainset_info[3]
                        tn += val_trainset_info[4]
                    acc, tp, fp, fn, tn = acc/total, tp/total, fp/total, fn/total, tn/total
                    self.model.train()
                    self.logger.info("acc: {}, tp: {}, fp: {}, fn: {}, tn: {}".format(acc, tp, fp, fn, tn))
                    eval_score = acc
                    self.saver.append_val_acc(eval_score)
                    if eval_score > best_score:
                        best_score = eval_score
                        self.save_model(args)
                        wait = 0
                    elif eval_score >0:
                        break
        self.logger.info('End: The model is:', args.method)
        self.saver.save_middle()
        return
    def train_semi(self, labeled_train_loader:RSNDataloader, unlabeled_train_loader:RSNDataloader, batch_size=100, same_ratio=0.5):
        self.optimizer.zero_grad()

        batch_data_left, batch_data_right, batch_data_label = labeled_train_loader.next_batch(batch_size, same_ratio = same_ratio)
        batch_data_left = self.to_cuda(*batch_data_left)
        batch_data_right = self.to_cuda(*batch_data_right)
        batch_data_label = batch_data_label.to(self.device)

        prediction, left_word_emb, right_word_emd, encoded_l, encoded_r = self.model(*batch_data_left, *batch_data_right)
        loss = self.model.get_loss_and_emb(prediction, batch_data_label)
        loss += self.model.get_v_adv_loss(batch_data_left, batch_data_right, prediction, left_word_emb.size()) * self.lambda_s

        ul_batch_data_left, ul_batch_data_right = unlabeled_train_loader.next_batch_ul(batch_size)
        ul_batch_data_left = self.to_cuda(*ul_batch_data_left)
        ul_batch_data_right = self.to_cuda(*ul_batch_data_right)

        prediction, left_word_emb, right_word_emd, encoded_l, encoded_r = self.model(*ul_batch_data_left, *ul_batch_data_right)

        loss += self.model.get_cond_loss(prediction) * self.p_cond
        loss += self.model.get_v_adv_loss(ul_batch_data_left, ul_batch_data_right, prediction, left_word_emb.size()) * self.lambda_s * self.p_cond

        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item()
        

    ### eval test data
    def eval(self, args, eval_data, is_test=False):
        dataloader = eval_data.test_dataloader
        self.logger.info("")
        self.model.eval()
        self.logger.info("eval model...........")
        with torch.no_grad():
            pred = []
            hiddens = []
            true_label = []
            for it, sample in enumerate(tqdm(dataloader)):
                batch_word, batch_mask, batch_pos, label = self.to_cuda(*sample)
                true_label.append(label.item())
                vector = self.model.get_hiddens(batch_word.long(), batch_mask.long(), batch_pos.long())
                hiddens.append(vector)
            hiddens = torch.cat(hiddens, dim=0)

        labels = true_label
        data_to_cluster = hiddens
        if 'Louvain' in args.select_cluster:
            self.logger.info('-----Louvain Clustering-----')
            pred = Louvain_no_isolation(dataset=data_to_cluster, edge_measure=self.model.pred_X)
        elif 'HAC' in args.select_cluster:
            self.logger.info('-----HAC Clustering-----')
            pred = complete_HAC(dataset=data_to_cluster, HAC_dist=self.model.pred_X,k=len(list(set(labels))))
        elif 'kmean' in args.select_cluster:
            self.logger.info('-----Kmeans Clustering-----')
            kmeans = KMeans(n_clusters=eval_data.ground_truth_num, n_init=10, random_state=args.seed).fit(hiddens.cpu().numpy())
            pred = kmeans.labels_.tolist()
        if not is_test:
            results = clustering_score(labels, pred)
        else:
            results = clustering_score(eval_data.test_label_ids, pred)
        self.logger.info("cluster_eval: {}".format(results))
        if is_test:
            reduce_dim = get_dimension_reduction(hiddens.cpu().numpy(), args.seed)
            self.saver.features = hiddens.cpu().numpy()
            self.saver.reduce_feat = reduce_dim
            self.saver.results = results
            self.saver.pred = pred
            self.saver.labels = eval_data.test_labels #list([eval_data.label_list[idx] for idx in labels])
            # self.saver.samples = eval_data.test_texts
            self.saver.known_label_list = eval_data.known_label_list
            self.saver.all_label_list = eval_data.all_label_list
            self.saver.save_output_results()
            results['B3'] = round(results['B3']['F1']*100, 2)
            self.saver.save_results(args, results)
        self.model.train()
        return results

    def restore_model(self, args):
        ckpt = restore_model(args)
        self.model.load_state_dict(ckpt['state_dict'])
    def save_model(self, args):
        save_dict = {'state_dict': self.model.state_dict()}
        save_model(args, save_dict)
