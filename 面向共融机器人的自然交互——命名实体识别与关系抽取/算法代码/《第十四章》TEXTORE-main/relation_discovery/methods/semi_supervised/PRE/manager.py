from tools.utils import *
from .memo import ODCMemory
from .loss import MetricLoss
from .model import Encoder

class Manager(BaseManager):
    def __init__(self, args, data = None):
        super(Manager, self).__init__(args)
        
        self.model = Encoder(args).cuda()
        self.model.to(self.device)
        # self.with_bert = True if args.backbone in ['bert'] else False
        # self.with_bert = True
        # self.logger.info("with bert : {}".format(self.with_bert))
        if args.freeze_bert_parameters:
            self.freeze_bert_parameters(self.model)
        self.metric_loss_unlabel = MetricLoss(args, rel_nums=args.n_clusters).loss_dict[args.loss_type]
        self.metric_loss_label = MetricLoss(args, rel_nums= args.label_rel_nums).loss_dict[args.loss_type]
    def train(self, args, data):

        if args.use_pretrain:
            temp_name = [args.dataname, args.this_name, args.seed, args.lr, args.known_cls_ratio, args.labeled_ratio]
            weight_name = "weights_{}.pth".format(
                        "-".join([str(x) for x in temp_name])
                    )
            mid_dir = creat_check_path(args.save_path, "relation_detection", "DTGNS")
            detect_path = os.path.join(mid_dir, weight_name)
            self.restore_model(args, pretrain=detect_path)
            self.logger.info("before training eval test data....")
            self.eval(args, data, dataloader=data.test_dataloader)
            self.logger.info("before training end....")
        #     have_pre = os.path.isfile(args.pretrain_model_file)
        #     have_pre =False
        #     if have_pre:
        #         self.logger.info("load haved pretrain!")
        #         self.restore_model(args, pretrain=args.pretrain_model_file)
        #     else:
        #         self.logger.info("begin pretrain")
        #         self.labeled_train(args, data)
        #         self.restore_model(args, pretrain=args.pretrain_model_file)
        #     self.logger.info("before training eval test data....")
        #     self.eval(args, data, dataloader=data.test_dataloader)
        #     self.logger.info("before training end....")
            
        self.self_train(args, data)
        
    def self_train(self, args, data):
        train_iter=args.loop_nums * len(data.semi_feat) // args.train_batch_size
        unlabel_loader = data.semi_train_dataloader
        self.odc_optim = self.get_optimizer(args, self.model)
        # if self.with_bert:
        self.odc_scheduler = self.get_scheduler(
            args, self.odc_optim,
            warmup_step=int(train_iter*args.warmup_proportion), 
            train_iter=train_iter
        )
        best_validation_f1 = 0
        not_best = 0
        best_si = 0
        self.memo = ODCMemory(
            len(unlabel_loader.dataset), 
            args.z_dim,
            momentum=args.moment,
            num_classes = args.n_clusters,
            logger = self.logger
        )
        self.logger.info("init memory")
        self.get_init_state(unlabel_loader)
        self.memo.init_memory()
        # self.memo.update_labels()

        
        for i in range(args.loop_nums):
            loss = self.unlabel_loop(unlabel_loader, i)
            self.logger.info("Finished {} loop, loss is {}".format(i+1, round(loss, 6)))

            si_score = self.memo.si_score()
            self.logger.info("si score: {}".format(si_score))
            results = self.eval(args,data, dataloader=data.eval_dataloader)
            si_score = results['si']
            self.logger.info("F1 score: {}".format(si_score))
            if si_score > best_si:
                not_best = 0
                best_si = si_score
                self.save_model(args) 
            else:
                not_best +=1
                self.logger.info("wait === {}!".format(not_best))
                if not_best>=args.wait_patient:
                    self.logger.info("early stop!")
                    break
        return

        
    def unlabel_loop(self, dataloader, nums):
        totle_loss = 0.0
        print("")
        self.model.train()
        td = tqdm(dataloader, desc="Train-{}".format(nums))
        for it, (w,m,p, inds,_) in enumerate(td):
            w,m,p = self.to_cuda(w,m,p)
            #### 1. get pseudo label
            pred = self.memo.pred[inds]
            #### 2. update loss

            h= self.model(w,m,p)

            loss = self.metric_loss_unlabel(h, pred)

            self.odc_optim.zero_grad()
            loss.backward()
 
            nn.utils.clip_grad_norm_(self.model.parameters(), 1) 
            
            self.odc_optim.step()
            # if self.with_bert:
            self.odc_scheduler.step()
            
            # #### 3. update sentence memory
            # states = h.clone()
            # self.memo.update_samples_memory(inds, states.detach().cpu())

            # # # #### 4. update centroids memory
            # self.memo.update_centroids_memory()
                
            totle_loss += loss.item()
            td.set_postfix(loss = totle_loss/(it + 1))
            # if it % 50 == 0:
        self.get_init_state(dataloader)
        self.memo.init_memory()
        # self.memo.update_centroids_memory()
        # self.memo.update_labels()
        # self.memo.print_class()
        print("")
        return totle_loss/(it + 1)

    

    def labeled_train(self, opt, data):
        train_loader, val_loader = data.sup_train_dataloader, data.eval_dataloader
        # unlabel_loader = data.unlabel_loaders
        train_iter=opt.pre_loop_nums * len(data.train_feat) // opt.train_batch_size
        self.optimizer = self.get_optimizer(opt, self.model)
        self.scheduler = self.get_scheduler(
                opt,self.optimizer,
                warmup_step=int(train_iter*opt.warmup_proportion), 
                train_iter=train_iter
            )
        best_validation_f1 = 0
        not_best = 0
        
        for epoch in range(1, opt.pre_loop_nums+1):
            self.logger.info('pretrain------epoch {}------'.format(epoch))
            self.logger.info('max batch num to train is {}'.format(opt.batch_num))
            all_loss = 0.0
            iter_sample = 0.0
            self.model.train()
            td = tqdm(train_loader, desc="Train-{}".format(epoch))
            for i, (w,m,p, inds, batch_label) in enumerate(td):
                w,m,p, batch_label = self.to_cuda(w,m,p, batch_label)

                features = self.model(w,m,p)
                loss = self.metric_loss_label(features, batch_label )

                self.optimizer.zero_grad()
                # if loss.item() < 1e-9:
                #     continue
                loss.backward()
                self.optimizer.step()
                # if self.with_bert:
                self.scheduler.step()
                
                iter_sample += 1
                all_loss += loss.item()
                td.set_postfix(loss = all_loss/iter_sample)
            # clustering & validation
            res = self.eval(opt, data, dataloader=val_loader)
            

            two_f1 = res['B3']['F1']
            self.logger.info("F1 score : {}".format(two_f1))
            self.logger.info("si score : {}".format(res['si']))
            if two_f1 > best_validation_f1:  # acc
                not_best = 0
                self.save_model(opt, pretrain=1)
                best_validation_f1 = two_f1
            else:
                not_best += 1
                self.logger.info("wait = {}".format(not_best))
            if not_best >= opt.wait_patient:
                break
                    
                    
            # break

        print('End: The model is:', opt.method)
        return
    def get_init_state(self, train_data):
        self.model.eval()
        self.logger.info("get init state...")
        states = None
        t = tqdm(train_data)
        for w, m, p, inds, _ in t:
            w, m, p = self.to_cuda(w, m, p)
            states = self.model(w, m, p)
            self.memo.memo_smaple[inds] = states.detach().cpu()
        self.logger.info("finished init state!")
        print("")
        
        return states
    
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
                
                batch_word, batch_mask, batch_pos, ind, label = self.to_cuda(*sample)
                true_label.append(label.item())

                vector = self.model(batch_word, batch_mask, batch_pos)
        
                hiddens.append(vector.cpu())
            hiddens = torch.cat(hiddens, dim=0)

        labels = true_label
        num_classes = len(np.unique(labels))
        kmeans = KMeans(n_clusters=num_classes, n_init=10, random_state=args.seed).fit(hiddens.numpy())
        pred = kmeans.labels_.tolist()
        si = si_score(hiddens.cpu().numpy(), kmeans.labels_)
        results = clustering_score(labels, pred)
        results['si'] = si
        self.logger.info("cluster_eval: {}".format(results))
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

    def restore_model(self, args, pretrain=None):
        temp_name = [args.dataname, args.this_name, args.seed, args.lr, args.known_cls_ratio, args.labeled_ratio]
        weight_name = "weights_{}.pth".format(
                    "-".join([str(x) for x in temp_name])
                )
        mid_dir = creat_check_path(args.save_path, "relation_detection", "DTGNS")
        pretrain = os.path.join(mid_dir, weight_name)
        # self.restore_model(args, pretrain=detect_path)
        ckpt = restore_model(args, output_model_file=pretrain)
        self.model.load_state_dict(ckpt['model'])  # , strict=False

    def save_model(self, args, pretrain = None):
        save_dict = {'model': self.model.state_dict()}
        if pretrain is None:
            save_model(args, save_dict)
        else:
            save_pretrain_model(args, save_dict)
