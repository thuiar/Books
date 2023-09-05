from tools.utils import *
from .model import BertForRE
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
from .model import AutoEncoder, AdaptiveClustering

class Manager(BaseManager):
    """
    Refactor "SelfORE: Self-supervised Relational Feature Learning for Open Relation Extraction"
    code base on: https://github.com/THU-BPM/SelfORE
    """
    def __init__(self, args, data=None):
        super(Manager, self).__init__(args)
        setup_seed(args.seed)
        self.model = BertForRE(args).to(self.device)
        if args.freeze_bert_parameters:
            self.freeze_bert_parameters(self.model)
            self.need_fix = False
        else:
            self.need_fix = True
        self.epoch = args.epoch
        self.no_pretrain = True
        self.train_batch_size = args.train_batch_size
        cluster = args.cluster
        self.warm_up = args.warm_up
        if cluster == 'kmeans':
            self.pseudo_model = KMeans(n_clusters=args.n_clusters, random_state=args.seed)
            self.adacluster = False
        elif cluster == 'adpative_clustering':
            self.pseudo_model = AdaptiveClustering(args, input_dim=self.model.hidden_size, device=self.device)
            self.adacluster = True
        self.train_words = None
        self.train_masks = None
        self.train_pos = None
        self.lpn = 0

    def train(self, args, data=None):
        self.train_words, self.train_masks, self.train_pos, _ = data.processor.convert_features_to_tensor(data.train_dataloader.dataset.data)
        self.logger.info("training total numbers: {}".format(len(self.train_words)))
        self.logger.info("starting ...")
        self.best_score = 0.0
        for _ in tqdm(range(args.loop_num)):
            acc = self.loop(args)
            if acc > self.best_score:
                self.best_score = acc
                self.save_model(args)
            # else:
            #     break
        
        if self.adacluster:
            self.logger.info("save cluster model")
            self.pseudo_model.save_model(self.pseudo_model.path2)

    def loop(self, args):
        self.lpn += 1
        self.logger.info("=== Generating Pseudo Labels...")
        bert_embs = self.get_hidden_state()
        if isinstance(self.pseudo_model, AdaptiveClustering):
            if self.no_pretrain:
                if os.path.isfile(self.pseudo_model.path1):
                    self.logger.info("=== Load exiest auto.pth...")
                    self.pseudo_model.load_model(self.pseudo_model.path1)
                else:
                    self.logger.info("=== Pretrain Aotuencoder...")
                    self.pseudo_model.pretrain(bert_embs)
                    self.pseudo_model.load_model(self.pseudo_model.path1)
                    self.logger.info("=== Pretrain Aotuencoder finised...")
                self.no_pretrain = False
            
            pseudo_labels = self.pseudo_model.fit(bert_embs, seed=args.seed).labels_
        else:
            pseudo_labels = self.pseudo_model.fit(bert_embs).labels_
        self.logger.info("=== Generating Pseudo Labels Done")

        self.logger.info("=== Training Classifier Model...")
        acc = self.train_encoder(args, pseudo_labels)
        self.logger.info("=== In loop Num {}, acc: {}".format(self.lpn, acc))
        self.logger.info("=== In loop Num {}, best acc: {}".format(self.lpn, self.best_score))
        self.logger.info("=== Training Classifier Model Done")

        return acc

        

    def get_hidden_state(self):
        train_set = TensorDataset(self.train_words, self.train_masks, self.train_pos)
        self.model.eval()
        self.logger.info("get hidden state.....")
        states = None
        hiddens = []
        train_loader = DataLoader(
            train_set,
            sampler=SequentialSampler(train_set),
            batch_size=self.train_batch_size
        )
        for it, sample in enumerate(tqdm(train_loader)):
            batch_word, batch_mask, batch_pos = self.to_cuda(*sample)
            states = self.model.get_hidden_state(batch_word, batch_mask, batch_pos)
            hiddens.append(states.detach().cpu())
        hiddens = torch.cat(hiddens, dim=0)
        self.logger.info("finished!")
        self.logger.info("")
        self.model.train()
        return hiddens.numpy()
    def train_encoder(self, args, labels):
        
        labels = torch.tensor(labels).long()

        dataset = TensorDataset(self.train_words, self.train_masks, self.train_pos, labels)

        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size])
        self.train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=args.batch_size
        )

        self.validation_dataloader = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            batch_size=args.batch_size
        )

        self.optimizer = AdamW(self.model.parameters(), lr=1e-5, eps=1e-8)
        epochs = self.epoch
        total_steps = len(self.train_dataloader) * epochs
        if self.warm_up:
            warmup_step = len(self.train_dataloader)
        else:
            warmup_step = 0
        self.logger.info("warmup step: {}".format(warmup_step))
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, warmup_step, num_training_steps=total_steps)
        # seed_val = 42
        # random.seed(seed_val)
        # np.random.seed(seed_val)
        # torch.manual_seed(seed_val)
        # torch.cuda.manual_seed_all(seed_val)
        self.model.cuda()
        best_acc = 0
        for epoch_i in range(0, epochs):
            # if epoch_i < 3 and self.need_fix:
            #     for i, param in enumerate(self.model.encoder.bert.named_parameters()):
            #         param[1].requires_grad = False
            # else:
            #     self.need_fix = False
            #     for i, param in enumerate(self.model.encoder.bert.named_parameters()):
            #         param[1].requires_grad = True
            # if args.freeze_bert_parameters:
            #     self.freeze_bert_parameters(self.model)
            self.logger.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            best_acc = self.train_epoch(args, best_acc)
        return best_acc
    
        
    def train_epoch(self, args, best_acc):
        total_train_loss = 0
        self.model.train()
        t = tqdm(self.train_dataloader)
        for it, sample in enumerate(t):
            batch_word, batch_mask, batch_pos, label = self.to_cuda(*sample)
            self.model.zero_grad()
            loss, _ = self.model(batch_word, batch_mask, batch_pos, labels=label)
            total_train_loss += loss.item()
            t.set_postfix(val_loss = total_train_loss / (it + 1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
        avg_train_loss = total_train_loss / len(self.train_dataloader)
        self.saver.append_train_loss(avg_train_loss)
        self.logger.info("")
        self.logger.info("  Average training loss: {0:.2f}".format(avg_train_loss))
        self.logger.info("")
        self.logger.info("Running Validation...")
        self.model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        t = tqdm(self.validation_dataloader)
        for it, batch in enumerate(t):
            b_input_ids, b_input_mask, b_input_pos, b_labels = self.to_cuda(*batch)
            with torch.no_grad():
                loss, logits = self.model(b_input_ids,
                                         b_input_mask,
                                         b_input_pos,
                                         labels=b_labels)
            total_eval_loss += loss.item()
            t.set_postfix(val_loss = total_eval_loss / (it + 1))
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += self.flat_accuracy(logits, label_ids)
        avg_val_accuracy = total_eval_accuracy / \
            len(self.validation_dataloader)
        self.saver.append_val_acc(avg_val_accuracy)
        self.logger.info("")
        self.logger.info("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        avg_val_loss = total_eval_loss / len(self.validation_dataloader)
        self.logger.info("  Validation Loss: {0:.2f}".format(avg_val_loss))
        if avg_val_accuracy > best_acc:
            best_acc = avg_val_accuracy
        #     self.save_model(args)
        # else:
        #     self.save_model(self.model)
        return best_acc

    ### eval test data
    def eval(self, args, eval_data, dataloader = None, is_test=True):
        dataloader = eval_data.test_dataloader
        self.model.eval()
        self.logger.info("testting...........")
        with torch.no_grad():
            pred = []
            hiddens = []
            true_label = []
            for it, sample in enumerate(tqdm(dataloader)):
                batch_word, batch_mask, batch_pos, label = self.to_cuda(*sample)
                true_label.append(label.item())
                hidden = self.model.get_hidden_state(batch_word, batch_mask, batch_pos)
                # y = torch.max(logits, dim=1)[1]
                # pred.append(y.item())
                hiddens.append(hidden)
            hiddens = torch.cat(hiddens, dim=0)
        
        kmeans = KMeans(n_clusters=eval_data.ground_truth_num, n_init=10, random_state=args.seed).fit(hiddens.cpu().numpy())
        pred = kmeans.labels_.tolist()
        
        results = clustering_score(eval_data.test_label_ids, pred)
        self.logger.info("cluster_eval: {}".format(results))

        reduce_dim = get_dimension_reduction(hiddens.cpu().numpy(), args.seed)
        self.saver.features = hiddens.cpu().numpy()
        self.saver.reduce_feat = reduce_dim
        self.saver.results = results
        self.saver.pred = pred
        self.saver.labels = eval_data.test_labels #list([eval_data.label_list[idx] for idx in labels])
        # self.saver.samples = eval_data.test_texts
        self.saver.known_label_list = eval_data.known_label_list
        self.saver.all_label_list = eval_data.all_label_list
        self.saver.save_output_results(
            {
                'true_label': true_label
            }
        )
        results['B3'] = round(results['B3']['F1']*100, 2)
        self.saver.save_results(args, results)
        self.model.train()
        return results
    @staticmethod
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)
    def restore_model(self, args):
        ckpt = restore_model(args)
        self.model.load_state_dict(ckpt['state_dict'])
    def save_model(self, args):
        save_dict = {'state_dict': self.model.state_dict()}
        save_model(args, save_dict)
