from tools.utils import *
from .encoder import Encoder
from .memo import ODCMemory

class Manager(BaseManager):
    def __init__(self, args, data=None):
        super(Manager, self).__init__(args)
        self.dataset = None
        self.labeled = None
        self.cr_base = 0.01
        self.model = Encoder(args).to(self.device)
        if args.freeze_bert_parameters:
            self.freeze_bert_parameters(self.model)
        


    def get_init_state(self, train_data):
        self.model.eval()
        self.logger.info("get init memory...")
        states = None
        for it, sample in enumerate(tqdm(train_data)):
            batch_word, batch_mask, batch_pos, inds, label = self.to_cuda(*sample)
            states = self.model.get_hidden_state(batch_word, batch_mask, batch_pos)
            states = states / torch.norm(states, dim=1, keepdim=True)
            self.memo.memo_smaple[inds] = states.detach().cpu()
        self.logger.info("finished init memory!")
        print("")
        self.model.train()
        return states
    def train(self, args, data):
        train_loader = data.train_dataloader
        self.memo = ODCMemory(
            len(train_loader.dataset),
            args.z_dim,
            momentum=args.moment,
            num_classes = data.n_clusters,
            logger = args.logger
        )
        #### init
        self.logger.info("init memory")
        self.get_init_state(train_loader)
        kmeans = KMeans(data.n_clusters, random_state=args.seed)
        predict = kmeans.fit_predict(self.memo.memo_smaple.numpy())
        self.memo.pred = torch.Tensor(predict).long().to(self.device) # init pseudo label
        self.memo.print_class()
        self.memo.update_centroids_memory(cinds=[x for x in range(args.n_clusters)])

        # self.odc_optim = self.get_optimizer(args, self.model)
        # self.odc_sche = self.get_scheduler(args, self.odc_optim
        # )
        train_iter=args.num_train_epochs * len(data.semi_feat) // args.train_batch_size
        self.odc_optim = self.get_optimizer(args, self.model)
        self.odc_scheduler = self.get_scheduler(
            args, self.odc_optim,
            warmup_step=int(train_iter*args.warmup_proportion), 
            train_iter=train_iter
        )

        ### labeled
        best_si = -1e3
        best_cr = 1
        wait = 0
        for i in range(args.num_train_epochs):
            loss = self.odsc_loop(train_loader, i)
            self.saver.append_train_loss(loss)
            self.logger.info("Finished {} loop, loss is {}".format(i+1, round(loss, 6)))
            si_score = self.memo.si_score()
            self.saver.append_train_acc(si_score)
            cr = self.memo.change_ratio()
            self.logger.info("si score: {}".format(si_score))
            self.logger.info("cr ratio: {}".format(cr))
            if si_score > best_si:
                best_si = si_score
                self.save_model(args)
                wait = 0
            else:
                wait +=1
                self.logger.info("wait{0}{1}".format('-'*10, wait))
                if wait >=args.wait_patient:
                    break
            if cr<self.cr_base:
                break
        self.saver.save_middle()
        return best_si
    def odsc_loop(self, dataloader, nums=0):
        ### update odsc
        totle_loss = 0.0
        print("")
        td = tqdm(dataloader, desc="Train-{}".format(nums))
        self.memo.old_pred = self.memo.pred.clone()
        for it, sample in enumerate(td):
            batch_word, batch_mask, batch_pos, inds, label = self.to_cuda(*sample)
            #### 1. get pseudo label
            pred = self.memo.pred[inds]
            #### 2. update loss
            logits, h = self.model(batch_word, batch_mask, batch_pos)
            loss = F.cross_entropy(logits, pred) # unlabel loss
            # self.warmup(self.odc_optim, warmup_step=500)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.odc_optim.step()
            self.odc_scheduler.step()
            self.odc_optim.zero_grad()
            
            #### 3. update sentence memory
            states = h.clone()
            cr = self.memo.update_samples_memory(inds, states.detach().cpu())
            #### 4. update centroids memory
            self.memo.update_centroids_memory()
            
                
            totle_loss += loss.item()
            td.set_postfix(loss = totle_loss/(it + 1))
        
        self.memo.deal_with_small_clusters()
        self.model.set_reweight(self.memo.pred.cpu().numpy())
        self.memo.print_class()
        print("")
        return totle_loss/(it + 1)
        # return self.memo.change_ratio()
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
                batch_word, batch_mask, batch_pos, inds, label = self.to_cuda(*sample)
                true_label.append(label.item())
                logits, hidden = self.model(batch_word, batch_mask, batch_pos)
                y = torch.max(logits, dim=1)[1]
                pred.append(y.item())
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

    def restore_model(self, args):
        ckpt = restore_model(args)
        self.model.load_state_dict(ckpt['state_dict'])
    def save_model(self, args):
        save_dict = {'state_dict': self.model.state_dict()}
        save_model(args, save_dict)
