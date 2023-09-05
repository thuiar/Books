# return data
from math import log
import re
from django.shortcuts import render
# download file
from django.http import FileResponse , JsonResponse
# return html
from django.views.decorators.clickjacking import xframe_options_exempt
# return Json , csv , shlex , subprocess , logging , os , sys , platform , shutil , stat , base64
import json , csv , shlex , subprocess , logging , os , sys , platform , shutil , stat , base64
from django.views.decorators.csrf import csrf_exempt
# time
from django.utils import timezone
from django.core.paginator import Paginator
from info.info_interaction import infoin, load_pickle, toclass


@xframe_options_exempt
def model_management(request):
    return render(request,'detection/model-list.html')

@csrf_exempt
def getModelList(request):
    model_name_select = request.GET.get('model_name_select')
    page = request.GET.get('page')
    limit = request.GET.get("limit")
    if model_name_select == None:
        model_name_select = ''
    modelList = infoin.model_tdes(task_type=1)
    modelList = modelList.loc[modelList['model_name'].str.contains(model_name_select)]
    modelList = infoin.convert_df_to_list(modelList)
    count = len(modelList)
    paginator = Paginator(modelList, limit)
    modelList = paginator.get_page(page)
    result = {}
    result['code'] = 0
    result['msg'] = ''
    result['count'] = count
    result['data'] = list(modelList)
    
    return JsonResponse(result)

@xframe_options_exempt
def model_management_details(request):
    model_id = request.GET.get('model_id')
    paramList, obj = infoin.hyper_parameters(model_id, task_type=1, toc=True)
    return render(request,'detection/model-details.html',{'obj':obj,'paramList':paramList})


@xframe_options_exempt
def model_training(request):
    return render(request,'detection/model-training-log-list.html')


@xframe_options_exempt
def getModelLogList(request):                                     
    type_select = request.GET.get('type_select')
    dataset_select = request.GET.get("dataset_select")
    model_select = request.GET.get("model_select")
    page = request.GET.get('page')
    limit = request.GET.get("limit")
    logList = infoin.run_log()
    modelList = infoin.model_tdes(task_type=1)
    if dataset_select == None:
        dataset_select = ''
    if model_select == None:
        model_select = ''
    logList = logList.loc[ logList["model_name"].isin(modelList["model_name"]) ]
    logList = logList.loc[logList["dataset_name"].str.contains(dataset_select) & logList["model_name"].str.contains(model_select)]
    if type_select != '5':
        logList = logList[logList["type"]==type_select]
    
    logList = infoin.convert_df_to_list(logList)
    paginator = Paginator(logList, limit)
    logList = paginator.get_page(page)
    count = paginator.count

    result = {}
    result['code'] = 0
    result['msg'] = ''
    result['count'] = count
    result['data'] = list(logList)
    return JsonResponse(result)

@xframe_options_exempt
def toLogParameter(request):
    log_id = request.GET.get('log_id')
    paramList = infoin.get_run_log_hyper_parameters(log_id=log_id)
    paramList = infoin.convert_df_to_classlist(paramList)
    ex_data = infoin.run_log()
    ex_data = ex_data.loc[ex_data["log_id"]==int(log_id)]
    ex_data = infoin.convert_df_to_classlist(ex_data)

    return render(request,'detection/model-training-log-parameter.html',{'model_id': log_id,'paramList':paramList,'ex_data':ex_data})

@xframe_options_exempt
def toRunModel(request):
    modelList = infoin.model_tdes(task_type=1)
    modelList = infoin.convert_df_to_classlist(modelList)
    datasetList = infoin.dataset()
    datasetList = infoin.convert_df_to_classlist(datasetList)
    result = {}
    result['modelList'] = modelList
    result['datasetList'] = datasetList
    
    return render(request,'detection/model-training-log-torun.html', result)

@csrf_exempt
def getParamListByModelId(request):
    model_id_select = request.GET.get('model_select')
    if model_id_select == None:
        return JsonResponse({'code':0,'msg':'','count':0,'data':[]})
    resultList, _ = infoin.hyper_parameters(int(model_id_select), task_type=1)
    paginator = Paginator(resultList, 100)
    resultList = paginator.get_page(1)

    result = {}
    result['code'] = 0
    result['msg'] = ''
    result['count'] = paginator.count
    result['data'] = list(resultList)
    return JsonResponse(result)


@csrf_exempt
def add_model_training_log(request):   

    print('there is add_model_training_log')
    model_id = request.POST['model_id']
    dataset_name_select = request.POST['dataset_name_select']
    Known_Ratio = request.POST['Known_Relation_Ratio']
    Annotated_Ratio = request.POST['Annotated_Ratio']
    params = request.POST['params']
    paramsListJson = json.loads(params)

    modelList = infoin.model_tdes(task_type=1)
    modelName = modelList.loc[modelList["model_id"]==int(model_id)]["model_name"].values[0]
    args = {
        "dname": dataset_name_select,
        "method": modelName,
        "task_type":'relation_detection',
        "kcr":Known_Ratio,
        "lab_ratio":Annotated_Ratio,
        "seed": 0,
        "istrain": 1
    }
    name = "{}_{}_{}_{}_{}.pkl".format(dataset_name_select.lower(), 0, "normal", Known_Ratio, float(Annotated_Ratio))
    mid_dir = os.path.join(infoin.result_path, "relation_detection", modelName)
    local_path = os.path.join(mid_dir, name)
    run_log, log_id = infoin.run_log(
        dataset_name = dataset_name_select, 
        model_name = modelName,
        model_id = model_id,
        Local_Path = local_path,
        create_time = timezone.now().strftime("%Y-%m-%d %H:%M:%S"),
        Annotated_ratio = Annotated_Ratio, 
        Known_Relation_Ratio = Known_Ratio,
        type = 1, # runing state
    )
    try:
        run_pid, process, type_after = infoin.exe(**args)
        run_log.loc[run_log["log_id"]==log_id, "run_pid"] = run_pid
        print(run_log)
        run_log.loc[run_log["log_id"]==log_id, "type"] = type_after
        print("run_ing:")
        print(type_after)
        run_log.to_csv(infoin.get_run_log_path(), index=False)
        infoin.run_log_hyper_parameters(paramsListJson, log_id)
    except :
        run_log.loc[run_log["log_id"]==log_id, "type"] = 3
        run_log.to_csv(infoin.get_run_log_path(), index=False)
        return JsonResponse({'code': 400, 'msg': 'Run  Process Has An Error ！！'})

    finally:
        pass

    print("Running finished!")

    return JsonResponse({'code':200,'msg':'Successfully Running Process!'})

@csrf_exempt
def kill_running(request):
    try:
        run_pid = request.POST.get('run_pid')
        log_id = request.POST.get('log_id')
        run_log = infoin.run_log()
        this_log = run_log.loc[run_log["log_id"]==int(log_id), "type"].values[0]
        if this_log != 1:
            return JsonResponse({'code': 400, 'msg': 'Process '+run_pid+' Was Over ！！'})
        command = 'kill -9 ' + str(run_pid)
        command = shlex.split(command)
        subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        run_log.loc[run_log["log_id"]==int(log_id), "type"] = 3
        run_log.loc[run_log["log_id"]==int(log_id), "Local_Path"] = ""
        run_log.to_csv(infoin.get_run_log_path(), index=False)

    except:
        return JsonResponse({'code': 400, 'msg': 'Kill  Process Has An Error ！！'})
    return JsonResponse({'code':200,'msg':'Successfully Kill Process!'})

@xframe_options_exempt
def model_test(request):

    datasetList = infoin.dataset()
    modelList_detection = infoin.model_tdes(task_type=1)
    df = infoin.run_log()
    if request.GET.get('log_id'):
        log_id = request.GET.get('log_id')
        cond = (df["log_id"]==int(log_id) )
        cond = cond & (df["type"]==2)
        create_time_new = df.loc[cond].tail(1)
    else:
        create_time_new = df.loc[(df["type"]==2) & (df["model_name"].isin(modelList_detection["model_name"].values.tolist()))]
        create_time_new = create_time_new.iloc[-1, :]
        log_id = create_time_new['log_id']
    if not isinstance(log_id, str):
        log_id = int(log_id)
    dataset_new = create_time_new['dataset_name']
    if not isinstance(dataset_new, str):
        dataset_new = dataset_new.values[0]
    model_new = create_time_new['model_name']
    if not isinstance(model_new, str):
        model_new = model_new.values[0]
    create_time = df.loc[(df['dataset_name']==dataset_new) & (df['model_name']==model_new) & (df['type']==2)]
    parameters = infoin.get_run_log_hyper_parameters(log_id=log_id)

    c_time = create_time_new['create_time']
    if not isinstance(c_time, str):
        c_time = c_time.values[0]
    tranf = infoin.convert_df_to_classlist
    result = {}
    result['datasetList'] = tranf(datasetList)
    result['modelList_detection'] = tranf(modelList_detection)
    result['create_time'] = tranf(create_time)
    result['create_time_new'] = tranf(create_time_new)
    result['parameters'] = tranf(parameters)
    result['show_selected'] = toclass({'dataname': dataset_new, 'modelname': model_new, \
        'createdtime': c_time, 'log_id': int(log_id)})
    
    return render(request,'detection/model-test.html' , result)


@csrf_exempt 
def check_evaluation(request):
    log_id= request.GET.get('log_id')
    modelList = infoin.run_log()
    modelList = infoin.convert_df_to_list(modelList.loc[modelList["log_id"]==log_id])
    result = list(modelList)

    return JsonResponse({'code':200,'msg':'Successfully !','data':result})

@csrf_exempt 
def show_create_time(request):
  
    dataset_name= request.GET.get('dataset_name')
    model_name= request.GET.get('model_name')
    if(dataset_name) == None:
        dataset_name_default = "Dataset"
    dataset_name_default =dataset_name
    if(model_name) == None:
        model_name_default = "Open Relation Detection"
    model_name_default = model_name
    run_log = infoin.run_log()
    create_time = run_log.loc[(run_log["dataset_name"]==dataset_name) & (run_log["model_name"]==model_name) & (run_log["type"]==2)]
    create_time = infoin.convert_df_to_list(create_time)

    datasetList = infoin.dataset()
    datasetList = infoin.convert_df_to_list(datasetList)
    modelList_detection = infoin.model_tdes(task_type=1)
    modelList_detection = infoin.convert_df_to_list(modelList_detection)
    result = {}
    result['datasetList'] = datasetList
    result['dataset_name_default'] = dataset_name_default
    result['modelList_detection'] = modelList_detection
    result['model_name_default'] = model_name_default
    result['create_time'] = create_time

    result = list(create_time)
    return JsonResponse({'code':200,'msg':'Successfully !','data':result})

@csrf_exempt 
def show_test_result(request):
    dataset_name = request.GET.get('dataset_name')
    method = request.GET.get('method')
    log_id = request.GET.get('log_id')#new
    return_list = infoin.get_test_results(log_id)
    if return_list is None:
        return JsonResponse({'code':200, 'msg':'There is no data', 'count':0, 'data':list([]) })
    results = {}
    results['data'] = return_list
    return JsonResponse({'code':200,'msg':'Successfully !','data':results})

def show_hyper_parameters(request):
    log_id= request.GET.get('log_id')
    parameters = infoin.get_run_log_hyper_parameters(log_id=log_id)
    parameters = infoin.convert_df_to_list(parameters)
    result = {}
    ratio=[]
    result = list(parameters)
    run_log = infoin.run_log()
    run_log = run_log.loc[run_log["log_id"]==int(log_id)]
    run_log = run_log.iloc[-1, :]
    ratio.append(float(run_log['Known_Relation_Ratio']))
    ratio.append(float(run_log['Annotated_ratio']))
    
    return JsonResponse({'code':200,'msg':'Successfully !','data':result,'ratio':ratio})
    
@csrf_exempt
def modelAnalysisTest(request):
    model_detection = request.POST['model_detection']
    example_select_detection = request.POST['example_select_detection']
    return JsonResponse({'code':200,'msg':'Successfully Detection The Relation!'})

@xframe_options_exempt
def model_analysis(request):#
    dataset_list = infoin.dataset()
    modelList_detection = infoin.model_tdes(task_type=1)
    run_log = infoin.run_log()
    if request.GET.get('log_id'):
        log_id = request.GET.get('log_id')
        run_log_select= run_log.loc[(run_log["log_id"]==int(log_id))&(run_log["type"]==2)]
        run_log_select = run_log_select.loc[run_log_select["model_name"].isin(modelList_detection["model_name"])]
        create_time_new = run_log_select.tail(1)
    else:
        run_log_select = run_log.loc[run_log["type"]==2]
        run_log_select = run_log_select.loc[run_log_select["model_name"].isin(modelList_detection["model_name"])]
        create_time_new = run_log_select.tail(1)
        log_id = create_time_new['log_id'].values[0]

    example_list = infoin.Model_Test_Example(log_id)
    dataset_new = create_time_new['dataset_name'].values[0]
    model_new = create_time_new['model_name'].values[0]
    create_time = run_log.loc[(run_log["dataset_name"]==dataset_new) & (run_log["model_name"]==model_new)]
    c_time = create_time_new['create_time']
    if not isinstance(c_time, str):
        c_time = c_time.values[0]
    trans = infoin.convert_df_to_classlist
    result = {}
    result['dataset_list'] = trans(dataset_list)
    result['modelList_detection'] = trans(modelList_detection)
    result['exampleList'] = trans(example_list)
    result['create_time'] = trans(create_time)
    result['create_time_new'] = trans(create_time_new)
    result['show_selected'] = toclass({'dataname': dataset_new, 'modelname': model_new, \
        'createdtime': c_time, 'log_id': int(log_id)})

    return render(request,'detection/model-analysis.html',result)


def test(request):
    a='okk'
    return JsonResponse(list(a),safe=False)

@csrf_exempt
def model_evaluation_getDataOfTFOverallByKey(request):
    key = request.GET.get('key')
    split_key = key.split('_')
    log_id = split_key[-1]
    res = infoin.get_true_flase(log_id)
    if res is None:
        return JsonResponse({ "code":201, "msg": "There is no data" })
    results = {}
    results['code'] = 200
    results['msg'] = ''
    results['data'] = res
    return JsonResponse(results)

@csrf_exempt
def model_evaluation_getDataOfTFFineByKey(request):
    key = request.GET.get('key')
    split_key = key.split('_')
    log_id = split_key[-1]
    res = infoin.get_true_flase(log_id, fine=True)
    if res is None:
        return JsonResponse({ "code":201, "msg": "There is no data" })
    results = {}
    results['code'] = 200
    results['msg'] = ''
    results['data'] = res
    return JsonResponse(results)

@csrf_exempt
def model_evaluation_getDataOfIOLRByKey(request):
    key = request.GET.get('key')
    split_key = key.split('_')
    log_id = split_key[-1]
    data = infoin.get_pickle_data(log_id)
    if data is None:
        return JsonResponse({ "code":201, "msg": "There is no data" })
    if len(data["train_loss"]) ==0 and len(data["train_acc"])==0:
        train_data = [0.0]
    elif len(data["train_loss"]) ==0:
        train_data = data["train_acc"]
    else:
        train_data = data["train_loss"]

    if len(data["val_loss"]) ==0 and len(data["val_acc"])==0:
        val_data = [0.0]
    elif len(data["val_loss"]) ==0:
        val_data = data["val_acc"]
    else:
        val_data = data["val_loss"]
    data_iokir = {
        "Training": train_data,
        "Validation":val_data
    }
    
    results = {}
    results['code'] = 200
    results['msg'] = ''
    results['data'] = data_iokir
    return JsonResponse(results)


@csrf_exempt
def model_analysis_getClassListByDatasetNameAndMethod(request):

    dataset_name = request.GET.get('dataset_name')
    method = request.GET.get('method')
    log_id = request.GET.get('log_id')
    class_type = 'known'
    page = request.GET.get('page')
    limit = request.GET.get("limit")
    return_list = infoin.get_test_example(log_id, dataset_name, method)
    if return_list is None:
        return JsonResponse({'code':200, 'msg':'There is no data', 'count':0, 'data':list([]) })
    
    count = len(return_list)
    paginator = Paginator(return_list, limit)
    return_list = paginator.get_page(page)

    results = {}
    results['code'] = 0
    results['msg'] = ''
    results['count'] = count
    results['data'] = list(return_list)
    return JsonResponse(results)

@csrf_exempt
def model_analysis_getTextListByDatasetNameAndMethodAndLabel(request):

    dataset_name = request.GET.get('dataset_name')
    method = request.GET.get('method')
    log_id = request.GET.get("log_id")
    label = request.GET.get('label_name')
    class_type = 'known'
    page = request.GET.get('page')
    limit = request.GET.get("limit")
    res = infoin.get_test_example(log_id, dataset_name, method, label)
    if res is None:
        return JsonResponse({'code':200, 'msg':'There is no data', 'count':0, 'data':list([]) })
    
    return_list = res
    
    count = len(return_list)
    paginator = Paginator(return_list, limit)
    return_list = paginator.get_page(page)

    results = {}
    results['code'] = 0
    results['msg'] = ''
    results['count'] = count
    results['data'] = list(return_list)
    return JsonResponse(results)


@csrf_exempt
def model_analysis_getDataByKey(request):
    
    key = request.GET.get('key')           
    split_key = key.split('_')
    log_id = split_key[-1]
    res = infoin.get_scatter_data(log_id)
    if res is None:
        return JsonResponse({ "code":201, "msg": "There is no data" })
    results = {}
    results['code'] = 200
    results['msg'] = ''
    results['data'] = res
    return JsonResponse(results)

def log_delete(request):
    log_id =  request.GET.get('log_id')
    run_log = infoin.run_log()
    ind = run_log.loc[run_log["log_id"]==int(log_id)].index.tolist()
    run_log = run_log.drop(ind)
    run_log.to_csv(infoin.get_run_log_path(), index=False)
    results = {}
    results['code'] = 200
    results['msg'] = 'del_okk'
 
    return JsonResponse(results)

