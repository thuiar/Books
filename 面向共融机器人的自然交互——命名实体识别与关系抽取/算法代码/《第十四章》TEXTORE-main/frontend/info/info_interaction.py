from genericpath import exists
import pandas as pd
import numpy as np
import os, yaml, json, pickle
import shlex, subprocess
import yaml
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment


class InfoInteraction(object):
    def __init__(self) -> None:
        super().__init__()
        self.current_path = os.getcwd()
        self.root = os.path.join(self.current_path, "frontend/info")
        self.config_path = self.current_path
        self.run_path = os.path.join(self.current_path, "run.py")
        self.result_path = os.path.join(self.current_path, "results")
        self.data_path = os.path.join(self.current_path, "data/demo_data")
        self.caches = {}
    def exe(self, dname=None, method=None, task_type=None, kcr=None, lab_ratio=None, seed=0, istrain=1):
        root = self.run_path
        eval_script = "python {} --dataname {} --method {} --task_type {}\
                    --known_cls_ratio {} --labeled_ratio {} --seed {}\
                    --train_model {} --gpu_id {}".format(root, dname, method, task_type, kcr, lab_ratio, seed, istrain, 0)
        eval_script = shlex.split(eval_script)
        eval_script[1] = root
        process = subprocess.Popen(eval_script)
        run_pid = process.pid
        # 1--runing  2--finished 3--failed
        process.communicate()
        run_type = process.returncode
        if run_type == 0 or run_type == '0':
            type_after = 2
        else :
            type_after = 3
        return run_pid, process, type_after
    def get_test_results(self, log_id):
        data = self.get_pickle_data(log_id)
        if data is None:
            return None
        return_list = data["results"]
        return return_list

    def get_info_from_path(self, results_path, keys):
        if not os.path.exists(results_path):
            ori = []
            df1 = pd.DataFrame(ori, columns=keys)
            df1.to_csv(results_path, index=False)
        data_diagram = pd.read_csv(results_path)
        return data_diagram
    def append_info_to_path(self, info:pd.DataFrame, results_path, keys, res):
        res = {k:res.get(k, "") for k in keys}
        df1 = pd.DataFrame([res], columns=keys)
        info = info.append(df1)
        info.to_csv(results_path, index=False)
        data_diagram = pd.read_csv(results_path)
        return data_diagram
    def dataset(self, **kwargs):
        keys = [
            "dataset_id",
            "dataset_name",
            "domain",
            "class_num",
            "source",
            "local_path",
            "type",
            "sample_total_num",
            "sample_training_num",
            "sample_validation_num",
            "sample_test_num",
            "sentence_max_length",
            "sentence_avg_length",
            "create_time"
        ]

        path = os.path.join(self.root, "datasets_info.yaml")
        info = load_yaml(path)
        # info = {}
        if len(kwargs) > 0:
            this_data = {"dataset_name": kwargs}
            new_info = info.update(this_data)
            save_yaml(path, new_info)
            info = new_info
        modelList = [v for k,v in info.items()]
        df = pd.DataFrame(modelList, columns=keys)
        df = df.fillna("")
        return df
    def del_dataset(self, dataset_name):
        keys = [
            "dataset_id",
            "dataset_name",
            "domain",
            "class_num",
            "source",
            "local_path",
            "type",
            "sample_total_num",
            "sample_training_num",
            "sample_validation_num",
            "sample_test_num",
            "sentence_max_length",
            "sentence_avg_length",
            "create_time"
        ]
        path = os.path.join(self.root, "datasets_info.yaml")
        info = load_yaml(path)
        info = {}
        info.pop(dataset_name)
        save_yaml(path, info)
        modelList = [v for k,v in info.items()]
        df = pd.DataFrame(modelList, columns=keys)
        df = df.fillna("")
        return df
    # data annotation 数据标注结果
    def annotation_result(self, ):
        keys = [
            "result_id",
            "dataset_id",
            "dataset_name",
            "sentences",
            "real_label",
            "predict_result",
            "candidate_label_1",
            "candidate_label_2",
            "candidate_label_3",
            "key_words",
            "create_time",
        ]
        pass
    # model management 管理模型运行代码信息
    def model_tdes(self, task_type = 1):
        keys = [
            "model_id",
            "model_name",
            "paper_source",
            "code_source",
            "local_path",
            "type",
            # "create_time",
        ]
        if task_type == 1:
            path = os.path.join(self.root, "detection_model_info.yaml")
        elif task_type == 2:
            path = os.path.join(self.root, "discovery_model_info.yaml")
        else:
            print("error")
        info = load_yaml(path)
        modelList = [v for k,v in info.items()]
        df = pd.DataFrame(modelList, columns=keys)
        return df
    def convert_df_to_list(self, df):
        df = [dict(x) for i, x in df.iterrows()]
        return df
    def convert_df_to_classlist(self, df):
        if isinstance(df, pd.DataFrame):
            df = self.convert_df_to_list(df)
        elif isinstance(df, pd.Series):
            df = [dict(df)]
        elif isinstance(df, dict):
            df = [df]
        df = [toclass(x) for x in df]
        return df
    # 模型超参
    def hyper_parameters(self, model_id, task_type=1, toc=False):
        keys = [
            "param_id",
            "param_name",
            "param_describe",
            "default_value",
            "value_type",
            "run_value",
            "model_id",
        ]
        modelList = self.model_tdes(task_type=task_type)
        modelinfo = modelList.loc[modelList["model_id"]==int(model_id)]
        modelName = modelinfo["model_name"].values[0]
        if task_type == 1:
            path = os.path.join(self.config_path, "relation_detection", "configs", modelName+".yaml")
        elif task_type == 2:
            path = os.path.join(self.config_path, "relation_discover", "configs", modelName+".yaml")
        else:
            print("error")
        this_params = load_yaml(path)
        all_params = []
        for k,v in this_params.items():
            params = {"param_name": k, "param_describe": v['desc'], "default_value": v['val'], "run_value": v['val']}
            if toc:
                all_params.append(toclass(params))
            else:
                all_params.append(params)
        modelinfo = self.convert_df_to_classlist(modelinfo)
        modelinfo = modelinfo[0]
        # this_params = self.convert_df_to_classlist(this_params)
        return all_params, modelinfo
    # 模型运行记录
    def run_log(self, **kwargs):
        path = os.path.join(self.root, "run_log.csv")
        keys = [
            "log_id",
            "dataset_name",
            "model_name",
            "model_id",
            "Annotated_ratio",
            "Known_Relation_Ratio",
            "Local_Path",
            "create_time",
            "type",
            "run_pid",
        ]
        info = self.get_info_from_path(path, keys)
        info = info.fillna("")
        if len(kwargs) > 0:
            log_id = len(info)
            if len(info)>0:
                log_id = info["log_id"].values.max() + 1
            kwargs["log_id"] = log_id
            info = self.append_info_to_path(info, path, keys, kwargs)
            return info, log_id
        return info
    def get_run_log_path(self):
        path = os.path.join(self.root, "run_log.csv")
        return path
    def get_run_log_hyper_parameters(self, log_id = None):
        path = os.path.join(self.root, "hyper_parameters.csv")
        keys = [
            "param_id",
            "param_name",
            "param_describe",
            "default_value",
            "value_type",
            "run_value",
            "log_id",
        ]
        info = self.get_info_from_path(path, keys)
        info = info.fillna("")
        if log_id is not None:
            parameters = self._cond_select(info, log_id=int(log_id))
            return parameters
        return info
    def run_log_hyper_parameters(self, res, log_id):
        path = os.path.join(self.root, "hyper_parameters.csv")
        keys = [
            "param_id",
            "param_name",
            "param_describe",
            "default_value",
            "value_type",
            "run_value",
            "log_id",
        ]
        info = self.get_info_from_path(path, keys)
        cur_len = len(info)
        new_res = []
        for i, item in enumerate(res):
            temp = {}
            temp["param_id"] = cur_len + i
            temp["param_name"] = item.get("param_name", "")
            temp["param_describe"] = item.get("param_describe", "")
            temp["default_value"] = item.get("default_value", "")
            temp["value_type"] = item.get("value_type", "")
            temp["run_value"] = item.get("run_value", "")
            temp["log_id"] = log_id
            new_res.append(temp)
        df1 = pd.DataFrame(new_res, columns=keys)
        info = info.append(df1)
        info.to_csv(path, index=False)
        return info
    def get_format_text(self, item:dict):
        text = item["text"]
        text = text.split(" ")
        h = item['h']['pos']
        t = item['t']['pos']
        if h[0] < t[0]:
            new_text = text[:h[0]] + ['[', '<font color="red">'] + text[h[0]:h[-1]] + ['</font>', ']<sub> e<sub>1</sub> </sub>'] + text[h[-1]:t[0]]\
                + ['[', '<font color="blue">'] + text[t[0]:t[-1]] + ['</font>', ']<sub> e<sub>2</sub> </sub>'] + text[t[-1]:]
        else:
            new_text = text[:t[0]] + ['[', '<font color="blue">'] + text[t[0]:t[-1]] + ['</font>', ']<sub> e<sub>2</sub> </sub>'] + text[t[-1]:h[0]]\
                + ['[', '<font color="red">'] + text[h[0]:h[-1]] + ['</font>', ']<sub> e<sub>1</sub> </sub>'] + text[h[-1]:]
        return new_text

    def _get_unique(self, ls:list):
        uni = []
        for u in ls:
            if u not in uni:
                uni.append(u)
        return uni
    
    def _get_open_text(self, dataname, class_type, method, this_text, keyword_or_label=None):
        # this_text: samples[i]
        
        text = this_text['text'].split(' ')
        new_text = self.get_format_text(this_text)
        tag = False if isinstance(keyword_or_label, str) else True
        if tag and class_type == 'open':
            lab_name0 = self.kw2list(keyword_or_label, tolist=True)
            lab_name = ", ".join([str(x) for x in lab_name0])
            temp = {
                    "dataset_name": dataname,
                    "class_type": class_type,
                    "label_name": lab_name,
                    "method": method,
                    "text": ' '.join(text),
                    "new_text": " ".join(new_text),
                    "can_1": lab_name0[0][0],
                    "can_2": lab_name0[1][0],
                    "can_3": lab_name0[2][0],
                    "conf_1": lab_name0[0][1],
                    "conf_2": lab_name0[1][1],
                    "conf_3": lab_name0[2][1],
                }
        else:
            temp = {
                        "dataset_name": dataname,
                        "class_type": class_type,
                        "label_name": keyword_or_label,
                        "method": method,
                        "text": ' '.join(text),
                        "new_text": " ".join(new_text)
                        }
        return temp

    
    def get_test_example(self, log_id, dataname, method, label=None, class_type='known'):
        data = self.get_pickle_data(log_id)
        if data is None:
            return None
        label_set = data["known_label_list"]
        labels = data["labels"]
        samples = data["samples"]
        pred = data["pred"]
        pred_set = self._get_unique(pred)
        if class_type == 'open':
            kw = data['keywords']
        res = []
        if label is not None:
            for i, p in enumerate(pred):
                if class_type == 'open':
                    lb = self.kw2list(kw[p])
                    gold_label = kw[p]
                else:
                    lb = p
                    gold_label = labels[i]
                if lb == label:
                    temp = self._get_open_text(
                            dataname,
                            class_type,
                            method,
                            samples[i],
                            keyword_or_label= gold_label
                        )
                    res.append(temp)
            return res
        for lb in pred_set:
            num = len([x for x in pred if x == lb])
            if class_type=='known':
                lab_name = lb
            else:
                lab_name = self.kw2list(kw[lb])
            temp = {
                "label_name": lab_name,
                "label_text_num": num,
                "dataset_name": dataname,
                "method": method,
                "class_type": class_type
            }
            res.append(temp)
        return res
    def kw2list(self, lab_name, tolist=False):
        # lab_name = kw[p]
        lab_name0 = [(x[0], str(round(x[1]*100, 2))+'%') for x in lab_name]
        if tolist:
            return lab_name0
        lab_name = ", ".join([str(x) for x in lab_name0])
        return lab_name
    def Model_Test_Example(self, log_id):
        keys = [
            "example_id",
            "sentences",
            "ground_truth",
            "predict_result",
            "candidate_label_1",
            "candidate_label_2",
            "candidate_label_3",
            "key_words",
            "type",
        ]
        data = self.get_pickle_data(log_id)
        if data is None:
            return None
        samples = data["samples"]
        label = data["labels"]
        pred = data["pred"]
        res = []
        for i in range(len(samples)):
            temp = {
                "example_id": i,
                "sentences": samples[i],
                "ground_truth": label[i],
                "predict_result": pred[i],
                "candidate_label_1": "label_1",
                "candidate_label_2": "label_2",
                "candidate_label_3": "label_3",
                "key_words": "None",
                "type": 11
            }
            if "keywords" in data:
                if not isinstance(data["keywords"], str):  
                    kw = data["keywords"][pred[i]]
                    temp["candidate_label_1"] = kw[0]
                    temp["candidate_label_2"] = kw[1]
                    temp["candidate_label_3"] = kw[2]
            res.append(temp)
        return res
        
    def Data_Note_Annotation_Result(self):
        keys = [
            "result_id",
            "dataset_id",
            "dataset_name",
            "method",
            "sentences",
            "real_label",
            "predict_result",
            "candidate_label_1",
            "candidate_label_2",
            "candidate_label_3",
            "key_words",
            "create_time",
        ]
    def get_pickle_data(self, log_id):
        run_log = self.run_log()
        path = self._cond_select(run_log, log_id=int(log_id))["Local_Path"].values[0]
        print(path)
        if not os.path.isfile(path):
            return None
        else:
            if path in self.caches:
                data = self.caches[path]
            else:
                data = load_pickle(path)
                self.caches[path] = data
        return data
    def get_true_flase(self, log_id, fine=False, tt=1):
        data = self.get_pickle_data(log_id)
        if data is None:
            return None
        return self.get_cm_data(data, fine, task_type=tt)
        
    def aligment_label(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D))
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1

        _, ind = linear_sum_assignment(w.max() - w)
        alignment_labels = list(ind)
        aligment2label = {label:i for i,label in enumerate(alignment_labels)}
        new_labels = np.array([aligment2label[label] for label in y_pred])
        return new_labels

    
    def get_cm_data(self, data, fine=False, task_type=1):
        pred = data['pred']
        label = data['labels']
        if task_type == 1:
            known_list = data["known_label_list"] + ['UNK']
            y_true = label
        elif task_type == 2:
            known_list = data["all_label_list"]
            labelmap = {y:i for i, y in enumerate(known_list)}
            maplabel = {k:v for v,k in labelmap.items()}
            if "y_true" in data:
                y_true = data["y_true"]
            else:
                
                y_true = [labelmap[x] for x in label]
            pred = self.aligment_label(y_true, pred)
            print(np.unique(y_true))
            print(np.unique(pred))
            pred = pred.tolist()
            pred = [maplabel[x] for x in pred]
        name = known_list
        print(name)
        cm = confusion_matrix(label, pred, labels = name)
        print(cm)
        
        d = {}
        if fine:
            for i in range(len(cm)):
                d[name[i]] = cm[i].tolist()
            return d
        d["relation_class"] = known_list
        is_true = []
        is_false = []
        for i in range(len(cm)):
            cur = name[i]
            cur_cm = cm[i]
            is_true.append(cur_cm[i])
            is_false.append(sum(cur_cm) - cur_cm[i])
        d["right"] = is_true
        d["left"] = [-1*x for x in is_false]
        d = str(d)
        d = eval(d)
        return d
    def get_scatter_data(self, log_id, task_type = 1):
        data = self.get_pickle_data(log_id)
        if data is None:
            return None
        tz = data["reduce_feat"]
        samples = data["samples"]
        label = data["labels"]
        pred = data["pred"]
        if isinstance(pred, np.ndarray):
            pred = pred.tolist()
        pred_set = self._get_unique(pred)
        if task_type == 2:
            kw = data["keywords"]
        points = {}
        texts = {}
        for i, kl in enumerate(pred_set):
            for j in range(len(pred)):
                if pred[j] == kl:
                    ttt = self.get_format_text(samples[j])
                    ttt = " ".join(ttt)
                    if kl in points:
                        points[kl].append(tz[j].tolist()+[ttt])
                        texts[kl].append(ttt)
                    else:
                        points[kl] = [tz[j].tolist()+[ttt]]
                        texts[kl] = [ttt]
        center_points = {}
        for i, kl in enumerate(pred_set):
            for j in range(len(pred)):
                if pred[j] == kl:
                    if kl in center_points:
                        center_points[kl].append(tz[j])
                    else:
                        center_points[kl] = [tz[j]]
        for k in pred_set:
            dt = np.stack(center_points[k], axis=0)
            dt = np.mean(dt, axis=0)
            if task_type == 2:
                lb = self.kw2list(kw[k], tolist=True)
                center_points[k] = [dt.tolist() + lb]
            else:
                center_points[k] = [dt.tolist() + [k]]
        res = {
            "points":points,
            "texts":texts,
            "center_points":center_points
        }
        new_data = {}

        # for i, kl in enumerate(pred_set):
        #     for j in range(len(pred)):
        #         if pred[j] == kl:
        #             x, y = tz[j].tolist()
        #             temp = [x,
        #                     y,
        #                     pred[j],
        #                     samples[j]["text"],
        #                     " ".join(samples[j]['h']['e1'][:-1]),
        #                     " ".join(samples[j]['t']['e2'][:-1])
        #                 ]
        #             if kl in new_data:
        #                 new_data[kl].append(temp)
        #             else:
        #                 new_data[kl] = [temp]
            
        # with open("/home/jaczhao/demo/ORE/frontend/info/scatter.json", 'w') as f:
        #     json.dump(new_data, f, indent=4)
        return res
    def _cond_select(self, df, **cond):
        for k,v in cond.items():
            df = df.loc[df[k] == v]
        return df
    def _get_pip_data(self, dataset_name, detection_method, discovery_method, know_relation, labeled_ratio, is_detect=True):
        run_log = self.run_log()
        run_log_select = self._cond_select(
            run_log,
            dataset_name=dataset_name,
            model_name = detection_method if is_detect else discovery_method,
            Annotated_ratio = float(labeled_ratio),
            Known_Relation_Ratio = float(know_relation),
            type=2
        )
        if len(run_log_select) <1:
            return None
        run_log_select = run_log_select.tail(1)
        log_id = run_log_select["log_id"].values[0]

        data = infoin.get_pickle_data(log_id)
        return data
    def get_pip_datalist(self, dataset_name, detection_method, discovery_method, know_relation, labeled_ratio):
        data = self._get_pip_data(dataset_name, detection_method, discovery_method, know_relation, labeled_ratio)
        if data is None:
            return None
        know = 0
        unknown = 0
        open_set = []
        labels = data["labels"]
        known = data["known_label_list"]
        all_label = data["all_label_list"]
        for lb in labels:
            if lb in known:
                know += 1
            else:
                unknown += 1
        res = {
            'dataset_name': dataset_name,
            'known_num': know,
            'unknown': unknown,
            'open': len(all_label) - len(known)
        }
        return res
    def get_pipclasslist_by_key(self, dataset_name, detection_method, discovery_method, know_relation, labeled_ratio, class_type='open', this_label=None):
        data = self._get_pip_data(dataset_name, detection_method, discovery_method, know_relation, labeled_ratio, is_detect=class_type == 'known')
        if data is None:
            return None
        pred = data['pred']
        if isinstance(pred, np.ndarray):
            pred = pred.tolist()
        pred_set = self._get_unique(pred)
        labels = data['labels']
        samples = data['samples']
        if class_type == 'open':
            kw = data["keywords"]
        res = []
        if this_label is not None:
            if this_label == 'unknown' or this_label == 'known':
                if this_label == 'unknown':
                    known_label = data["known_label_list"]
                else:
                    known_label = [x for x in data["all_label_list"] if x not in data["known_label_list"]]

                for i, lb in enumerate(labels):
                    if lb not in known_label:
                        temp = self._get_open_text(
                            dataset_name,
                            class_type,
                            detection_method,
                            samples[i],
                            labels[i]
                        )
                        res.append(temp)
                return res
            else:
                for p in pred_set:
                    if class_type == 'open':
                        lb = self.kw2list(kw[p])
                    else:
                        lb = p
                    if lb == this_label:
                        for i, pp in enumerate(pred):
                            if pp==p:
                                temp = self._get_open_text(
                                    dataset_name,
                                    class_type,
                                    discovery_method if class_type=='open' else detection_method,
                                    samples[i],
                                    kw[p] if class_type == 'open' else labels[i]

                                )
                                res.append(temp)
            return res
        for p in pred_set:
            
            if class_type == 'open':
                lb_name = self.kw2list(kw[p])
            else:
                if p == 'UNK':
                    continue
                lb_name = p
            this_p = [x for x in pred if x == p]
            res.append({
                'label_name': lb_name,
                'label_text_num': len(this_p),
                'dataset_name': dataset_name,
                'method': discovery_method if class_type=='open' else detection_method,
                'class_type': class_type
            })

        return res

    
def save_yaml(path:str, d:dict):
    this_d = {}
    for k, v in d.items():
        if isinstance(v, str) or isinstance(v, int) or isinstance(v, float):
            this_d[k] = v
    with open(path, 'w') as file: 
        yaml.dump(this_d, file, default_flow_style=False)

def load_yaml(path:str, args=None):
    with open(path, 'r') as f:
        d = yaml.load(f, Loader=yaml.FullLoader)
        if args is not None:
            for key, value in d.items():
                vars(args)[key] = value
    return d

class toclass(object):
    def __init__(self, a:dict):
        for k, v in a.items():
            setattr(self, k, v)

def save_pickle(path:str, d):
    with open(path, 'wb') as f:
        pickle.dump(d, f)

def load_pickle(path:str):
    import torch
    with open(path, 'rb') as f:
        d = pickle.load(f)
    tag = False
    for k,v in d.items():
        if isinstance(v, torch.Tensor):
            tag = True
            try:
                d[k] = v.cpu().numpy()
            except:
                d[k] = v.numpy()
    if tag:
        save_pickle(path, d)
    return d

infoin = InfoInteraction()