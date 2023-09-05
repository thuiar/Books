from relation_discover.utils import *
from .encoder import Encoder
class Manager(BaseManager):
    def __init__(self, args):
        super(Manager, self).__init__()
        self.model = Encoder(args).cuda()
        self.pos_margin = args.pos_margin 
        self.neg_margin = args.neg_margin 
        self.temperature = args.temp 
        self.states = None
        self.num_classes =  args.n_clusters

    def train(self, args, data):
        have_pre = get_pretrain(args)
        if have_pre:
            logger.info("Load the existing pretrain weights!")
            self.restore_model(args, pretrain=args.pretrain_name)
        else:
            logger.info("begin pretrain")
            pre_train = PreTrain(args)
            pre_train.labeled_train(args, data)
            del pre_train
            self.restore_model(args, pretrain=args.pretrain_name)
        print("before training eval test data....")
        self.eval(args, eval_data=data.test_data_loader)
        print("before training end....")
        
        self.self_train(args, data)
    
    def self_train(self, args, data):
        unlabel_loader = data.unlabel_loader
        args.train_iter = args.loop_nums * len(unlabel_loader.dataset)
        args.warmup_step = args.warmup_ratio * args.train_iter
        self.odc_optim = self.get_optimizer(args, self.model)
        self.odc_scheduler = self.get_scheduler(args, self.odc_optim)
        not_best = 0
        best_f1 = 0
        self.length = len(unlabel_loader.dataset)
        self.hidden_size = args.z_dim
        
        for i in range(args.loop_nums):
            loss = self.unlabel_loop(unlabel_loader, i)
            logger.info("Finished {} loop, loss is {}".format(i+1, round(loss, 6)))
            results = self.eval(args, eval_data=data.eval_loader)
            f1_score = results['F1']
            
            logger.info("F1 score: {}".format(f1_score))
            if f1_score > best_f1:
                not_best = 0
                best_f1 = f1_score
                self.save_model(args) 
            else:
                not_best +=1
                logger.info("wait === {}!".format(not_best))
                if not_best>=args.wait_patient:
                    logger.info("early stop!")
                    break
        return

    def unlabel_loop(self, dataloader, nums):
        totle_loss = 0.0
        self.get_init_state(dataloader)
        kmeans = KMeans(self.num_classes, n_init=10)
        predict = kmeans.fit_predict(self.states.numpy())

        kk = np.bincount(np.array(predict), minlength=self.num_classes)
        logger.info("class count: {}".format(kk))
        pseudo_label = torch.Tensor(predict).long().cuda() # init pseudo label
        print("")
        self.model.train()
        td = tqdm(dataloader, desc="Train-{}".format(nums))
        for it, (batch_data, inds) in enumerate(td):
            batch_data = self.to_cuda(batch_data)
            #### 1. get pseudo label
            pred = pseudo_label[inds]
            #### 2. update loss
            h = self.model(batch_data)
            loss = self.model.loss(h, pred, unlabel=True)
            self.odc_optim.zero_grad()
            loss.backward()
            self.odc_optim.step()
            self.odc_scheduler.step()
            totle_loss += loss.item()
            td.set_postfix(loss = totle_loss/(it + 1))
        
        print("")
        return totle_loss/(it + 1)
    # get hidden features
    def get_init_state(self, train_data):
        memo_smaple = torch.zeros((self.length, self.hidden_size))
        logger.info("get state")
        self.model.eval()
        t = tqdm(train_data)
        for batch_data, inds in t:
            batch_data = self.to_cuda(batch_data)
            states = self.model.get_h(batch_data)
            memo_smaple[inds] = states.detach().cpu()
        logger.info("finished init state!")
        print("")
        self.states = memo_smaple
        return memo_smaple
    ### eval test data
    def eval(self, args, eval_data):
        print("")
        self.model.eval()
        logger.info("eval model...........")
        with torch.no_grad():
            pred = []
            hiddens = []
            true_label = []
            for it, sample in enumerate(tqdm(eval_data)):
                batch_data, label = sample
                true_label.append(label.item())
                batch_data, label = self.to_cuda(batch_data, label)
                vector = self.model.get_h(batch_data)
                hiddens.append(vector)
            hiddens = torch.cat(hiddens, dim=0)

        labels = true_label
        num_classes = len(np.unique(labels))

        logger.info("nums claases: {}".format(num_classes))

        logger.info("-----Kmean Clustering-----")
        kmeans = KMeans(n_clusters=num_classes, random_state=100).fit(hiddens.cpu().numpy())
        pred = kmeans.labels_.tolist()
        cluster_eval_b3 = ClusterEvaluation(labels, pred).printEvaluation()
        logger.info("cluster_eval: {}".format(str(cluster_eval_b3)))
        NMI_score = normalized_mutual_info_score(labels, pred)
        cluster_eval_b3['nmi'] = NMI_score
        print("resluts {}".format(cluster_eval_b3))
        self.model.train()
        return cluster_eval_b3
    
    def change_ratio(self):
        a = (self.memo.pred.cpu() != self.old_pred.cpu()).sum().float() / self.memo.pred.shape[0]
        logger.info("change ratio: {}".format(round(a.numpy()*100, 4)))
        return a

    def restore_model(self, args, pretrain=None):
        ckpt = restore_model(args, pretrain_name=pretrain)
        state_dict = ckpt['state_dict']
        own_state = self.model.state_dict()
        print("load paprms")
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            try:
                own_state[name].copy_(param)
            except:
                print(name)
                print("")

    def save_model(self, args, pretrain = None):
        save_dict = {'state_dict': self.model.state_dict()}
        save_model(args, save_dict, pretrain_name=pretrain)

class PreTrain(BaseManager):
    def __init__(self, args):
        super(PreTrain, self).__init__()
        self.model = Encoder(args).cuda()
    def labeled_train(self, opt, data):
        train_loader, val_loader = data.train_data_loader, data.eval_loader
        opt.train_iter = opt.pre_loop_nums * opt.batch_num
        opt.warmup_step = opt.warmup_ratio * opt.train_iter
        self.optimizer = self.get_optimizer(opt, self.model)
        self.scheduler = self.get_scheduler(opt, self.optimizer)
        best_validation_f1 = 0
        not_best = 0
        for epoch in range(1, opt.pre_loop_nums+1):
            if not_best > opt.wait_patient:
                break
            logger.info('pretrain------epoch {}------'.format(epoch))
            logger.info('max batch num to train is {}'.format(opt.batch_num))
            all_loss = 0.0
            iter_sample = 0.0
            self.model.train()
            for i in range(1, opt.batch_num + 1):
                batch_data, batch_label = next(train_loader)
                batch_data, batch_label = self.to_cuda(batch_data, batch_label)

                loss = self.model(batch_data, batch_label, unlabel=False)

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
                    cluster_eval_b3 = self.eval(opt, val_loader)
                    

                    two_f1 = cluster_eval_b3['F1']
                    logger.info("F1 score : {}".format(cluster_eval_b3['F1']))
                    if two_f1 > best_validation_f1:  # acc
                        not_best = 0
                        self.save_model(opt, pretrain=opt.pretrain_name)
                        best_validation_f1 = two_f1
                    else:
                        not_best += 1
                        logger.info("wait = {}".format(not_best))
                        if not_best >= opt.wait_patient:
                            break
                    all_loss = 0.0
                    iter_sample = 0.0
        print('EndPretrian')
        return
    ### eval test data
    def eval(self, args, eval_data):
        print("")
        self.model.eval()
        logger.info("eval model...........")
        with torch.no_grad():
            pred = []
            hiddens = []
            true_label = []
            for it, sample in enumerate(tqdm(eval_data)):
                batch_data, label = sample
                true_label.append(label.item())
                batch_data, label = self.to_cuda(batch_data, label)
                vector = self.model(batch_data)
                hiddens.append(vector)
            hiddens = torch.cat(hiddens, dim=0)

        labels = true_label
        num_classes = len(np.unique(labels))

        logger.info("nums claases: {}".format(num_classes))
        logger.info("-----Kmean Clustering-----")
        kmeans = KMeans(n_clusters=num_classes, random_state=100).fit(hiddens.cpu().numpy())
        pred = kmeans.labels_.tolist()
        cluster_eval_b3 = ClusterEvaluation(labels, pred).printEvaluation()
        logger.info("cluster_eval: {}".format(str(cluster_eval_b3)))
        NMI_score = normalized_mutual_info_score(labels, pred)
        cluster_eval_b3['nmi'] = NMI_score
        print("resluts {}".format(cluster_eval_b3))
        self.model.train()
        return cluster_eval_b3
    def save_model(self, args, pretrain = None):
        save_dict = {'state_dict': self.model.state_dict()}
        save_model(args, save_dict, pretrain_name=pretrain)