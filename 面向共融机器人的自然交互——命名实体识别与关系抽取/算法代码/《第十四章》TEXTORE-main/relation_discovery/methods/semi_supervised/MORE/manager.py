from numpy.core.fromnumeric import nonzero
from tools.utils import *
from .loss import pairwise_distance

class Manager(BaseManager):
    def __init__(self, args, data=None):
        super(Manager, self).__init__(args)
        if 'cnn' in args.this_name:
            self.use_bert = False
            from .more_cnn import PCnn
            self.model = PCnn(args).cuda()
            self.optimizer = optim.RMSprop(self.model.get_params(args.learning_rate))
        else:
            self.use_bert = True
            from .more_bert import MORE_BERT
            self.model = MORE_BERT(args).cuda()
            if args.freeze_bert_parameters:
                self.freeze_bert_parameters(self.model)
            self.optimizer = self.get_optimizer(args, self.model)
            self.scheduler = self.get_scheduler(args, self.optimizer, warmup_step=500, train_iter=args.batch_num * args.loop_nums)
            # if args.freeze_bert_parameters:
            #     self.freeze_bert_parameters(self.model)

    def train(self, opt, data):
        train_loader, val_loader = data.sup_train_dataloader, data.eval_dataloader
        best_validation_f1 = 0
        not_best = 0
        
        for epoch in range(1, opt.loop_nums+1): # loop 4
            print('------epoch {}------'.format(epoch))
            print('max batch num to train is {}'.format(opt.batch_num))
            all_loss = 0.0
            iter_sample = 0.0
            self.model.train()
            if not_best > opt.wait_patient:
                break
            for i in range(1, opt.batch_num + 1):
                batch_word, batch_mask, batch_pos, batch_label = self.to_cuda(*next(train_loader))
                if self.use_bert:
                    features = self.model(batch_word, batch_mask, batch_pos)
                    loss = self.model.ml.cluster_loss(features, batch_label, opt)
                else:
                    wordembed, features = self.model.forward_norm(batch_word)
                    loss = self.model.ml.cluster_loss(features, batch_label, opt)
                    if opt.vat and epoch >= opt.warmup:#opt.warmup
                        vat_loss = self.vat_loop(opt, wordembed, features, batch_word)
                        loss += vat_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                iter_sample += 1
                all_loss += loss.item()
                sys.stdout.write('step: {0:4} | loss: {1:2.6f}'.format(i, all_loss / iter_sample) +'\r')
                sys.stdout.flush()
                # clustering & validation
                if i % opt.val_step == 0:

                    res = self.eval(opt, data, val_loader)
                    self.saver.append_train_loss(all_loss / iter_sample)
                    self.saver.append_val_acc(res['ACC'])
                    two_f1 = res['B3']['F1']
                    self.logger.info("B3 F1 : {}".format(two_f1))
                    if two_f1 > best_validation_f1:  # acc
                        not_best = 0
                        self.save_model(opt)
                        best_validation_f1 = two_f1
                    else:
                        # not_best += 1
                        # self.logger.info("wait---- {}".format(not_best))
                        # if not_best > opt.wait_patient:
                        break
                    
                    all_loss = 0.0
                    iter_sample = 0.0
        print('End: The model is:', opt.method)
        self.saver.save_middle()
        return
    def vat_loop(self, opt, wordembed, features, batch_data):
        
        disturb = self.to_cuda(torch.FloatTensor(wordembed.shape).uniform_(0, 1))[0]
        disturb.requires_grad = True

        disturb = (opt.p_mult) * F.normalize(disturb, p=2, dim=1)
        _, encode_disturb = self.model.forward_norm(batch_data, disturb)
        dist_el = torch.sum(torch.pow(features - encode_disturb, 2), dim=1).sqrt()
        diff = (dist_el / 2.0).clamp(0, 1.0 - 1e-7)
        disturb_gradient = torch.autograd.grad(diff.sum(), disturb, retain_graph=True)[0]
        disturb = disturb_gradient.detach()
        disturb = opt.p_mult * F.normalize(disturb, p=2, dim=1)

        # virtual adversarial loss
        _, encode_final = self.model.forward_norm(batch_data, disturb)

        # compute pair wise use the new embedding
        dist_el = torch.sum(torch.pow(features - encode_final, 2), dim=1).sqrt()
        diff = (dist_el / 2.0).clamp(0, 1.0 - 1e-7)
        v_adv_losses = torch.mean(diff)
        
        assert torch.mean(v_adv_losses).item() > 0.0
        return v_adv_losses * opt.lambda_V
    ### eval test data
    def eval(self, args, eval_data, dataloader = None, is_test=False):
        if dataloader is None:
            dataloader = eval_data.test_dataloader
        print("")
        self.model.eval()
        self.logger.info("eval model...........")
        with torch.no_grad():
            pred = []
            hiddens = []
            true_label = []
            for it, sample in enumerate(tqdm(dataloader)):
                
                batch_word, batch_mask, batch_pos, label = self.to_cuda(*sample)
                true_label.append(label.item())
                if self.use_bert:
                    vector = self.model.forward(batch_word, batch_mask, batch_pos)
                else:
                    _, vector = self.model.forward_norm(batch_word, batch_mask, batch_pos)  # [1,embedding_size]
                hiddens.append(vector.cpu())
            hiddens = torch.cat(hiddens, dim=0)

        labels = true_label
        # num_classes = len(np.unique(labels))
        if is_test:
            num_classes = eval_data.ground_truth_num
        else:
            num_classes = len(np.unique(labels))
        kmeans = KMeans(n_clusters=num_classes, n_init=10, random_state=args.seed).fit(hiddens.numpy())
        pred = kmeans.labels_.tolist()
        
        if not is_test:
            results = clustering_score(labels, pred)
        else:
            results = clustering_score(eval_data.test_label_ids, pred)
        if is_test:
            reduce_dim = get_dimension_reduction(hiddens.numpy(), args.seed)
            self.saver.features = hiddens.numpy()
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
