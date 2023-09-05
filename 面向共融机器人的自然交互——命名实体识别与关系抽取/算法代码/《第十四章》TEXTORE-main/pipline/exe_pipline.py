from tools.utils import *
from tools.config import Param

def get_final_results(args, task_type):
    """
    self.outputs = {
            "samples": self.samples,
            "labels": self.labels,
            "pred": self.pred,
            "features": self.features,
            "reduce_feat": self.reduce_feat,
            "results": self.results,
            "train_loss": self.train_loss, "train_acc": self.train_acc, "val_loss":self.val_loss, "val_acc":self.val_acc
        }
    """

    method = args.detection_method if task_type == 'relation_detection' else args.discovery_method
    mid_dir = creat_check_path(args.result_path, task_type, method)
    if args.method_type in ['unsupervised']:
        name = "{}_{}_{}_{}_{}.pkl".format(args.dataname, args.seed, args.this_name, 0, args.labeled_ratio)
    else:
        name = "{}_{}_{}_{}_{}.pkl".format(args.dataname, args.seed, args.this_name, args.known_cls_ratio, args.labeled_ratio)
    res_path = os.path.join(mid_dir, name)
    print("*"*20)
    print(res_path)
    print("*"*20)
    res = load_pickle(res_path)
    return res

def combine_test_results(args,  detect_res, discover_res, open_k_num = None, known_label_list=None):
    
    
    print(detect_res['known_label_list'])
    print(discover_res['known_label_list'])
    known_label_list = detect_res['known_label_list']
    # all_label_list = discover_res['all_label_list']
    map_list1 = {k:v for v,k in enumerate(known_label_list)}
    # map_list2 = {k:v for v,k in enumerate(all_label_list)}
    
    detection_preds = np.array(detect_res['pred'])
    discovery_feat = discover_res['features']
    dpreds = [str(x) for x in discover_res['pred']]
    discovery_preds = np.array(dpreds, dtype=detection_preds.dtype)
    if len(discover_res['labels']) < 1:
        path = os.path.join(args.data_dir, args.dataname, 'test.json')
        data = load_json(path)
        lb = []
        for item in data:
            lb.append(item['relation'])
        discover_res['labels'] = lb
    all_label_list = []
    for x in discover_res['labels']:
        if x not in all_label_list:
            all_label_list.append(x)
    map_list2 = {k:v for v,k in enumerate(all_label_list)}
    y_true =  np.array(discover_res['labels'])
    
    logger=args.logger

    unseen_token_id = "UNK"

    numpy_keys = ['y_pred', 'y_true', 'y_feat']
    # for key in numpy_keys:
    #     discovery_results[key] = np.array(discovery_results[key])

    pred_known_ids = [idx for idx, label in enumerate(detection_preds) if label != unseen_token_id]
    pred_open_ids = [idx for idx, label  in enumerate(detection_preds) if label == unseen_token_id]

    open_feats = discovery_feat[pred_open_ids]
    n_known_cls = len(known_label_list)
    num_labels = len(all_label_list)

    open_k_num = num_labels - n_known_cls

    km = KMeans(n_clusters=open_k_num, random_state = args.seed)
    km.fit(open_feats)
    # ###
    # km_all = KMeans(n_clusters=num_labels, n_jobs=-1, random_state = args.seed)
    # km_all.fit(discover_res['features'])
    # predsss = km_all.labels_
    # print("this_pre")
    # print(sum([int(x==y) for x,y in zip(predsss, discover_res['pred'])])/ float(len(predsss)))
    # ###

    open_labels = km.labels_ + n_known_cls
    open_labels_set = [str(x) for x in list(set(open_labels))]

    discovery_preds[pred_known_ids] = detection_preds[pred_known_ids]
    discovery_preds[pred_open_ids] = open_labels

    print(np.unique(discovery_preds))

    test_known_ids =  [idx for idx, label in enumerate(y_true) if label in known_label_list]

    known_true = y_true[test_known_ids]
    known_pred = discovery_preds[test_known_ids].copy()
    
    unseen_label_id = len(known_label_list)
    unseen_token = 'UNK'
    for idx, elem in enumerate(known_pred):
        if elem in open_labels_set:
            known_pred[idx] = unseen_token
    print(np.unique(known_pred))

    known_cm = confusion_matrix(known_true, known_pred)

    known_relation_acc = accuracy_score(known_true, known_pred)
    known_relation_f1 = F_measure(known_cm)['Overall']

    test_open_ids = [idx for idx, label in enumerate(y_true) if label not in known_label_list]
    
    open_true = y_true[test_open_ids]
    open_pred = discovery_preds[test_open_ids]

    open_cm = confusion_matrix(open_true, open_pred)

    open_true = [map_list2[x] for x in open_true]
    open_pred = [map_list1[x] if x not in open_labels_set else int(x) for x in open_pred]

    print(clustering_score([map_list2[x] for x in y_true], discover_res['pred']))
    
    print(clustering_score(open_true, [x for i,x in enumerate(discover_res['pred']) if i in test_open_ids]))

    open_relation_score = clustering_score(open_true, open_pred)

    open_relation_score['B3'] = round(open_relation_score['B3']['F1']*100, 2)

    open_relation_score['Known_Acc'] = known_relation_acc

    open_relation_score['Known_F1'] = known_relation_f1

    return open_relation_score


def run_pipline(args):

    detect_res = get_final_results(args, 'relation_detection')

    discover_res = get_final_results(args, 'relation_discovery')

    score = combine_test_results(args, detect_res, discover_res)

    print(score)
    # saver = SaveData(args, save_cfg=False)
    # args.this_name = 'pipeline'
    # saver.save_results(args, score, use_thisname=True, save_path="/home/jaczhao/OpenORE/from162/ORE/results/pipeline/")
    




    # print(confusion_matrix(detect_res['labels'], detect_res['pred']))
    # map_list = {}
    # i = 0
    # for l in discover_res['labels']:
    #     if l not in map_list:
    #         map_list[l] = i
    #         i += 1
    # discover_res['labels'] = [map_list[l] for l in discover_res['labels']]
    # print(confusion_matrix(discover_res['labels'], discover_res['pred']))




