# return data
from django.shortcuts import render
# download file
from django.http import FileResponse , JsonResponse
# return html
from django.views.decorators.clickjacking import xframe_options_exempt
# return Json , csv , shlex , subprocess , logging , os , sys , platform , shutil , stat , base64
import json , csv, os , sys , platform 
from django.views.decorators.csrf import csrf_exempt
# time
from django.utils import timezone
from django.core.paginator import Paginator

from info.info_interaction import infoin, load_pickle, toclass
 

@xframe_options_exempt
def model_management(request):
    return render(request,'discovery/model-list.html')

@csrf_exempt
def getModelList(request):
    model_name_select = request.GET.get('model_name_select')
    page = request.GET.get('page')
    limit = request.GET.get("limit")

    if model_name_select == None:
        model_name_select = ''
    modelList = infoin.model_tdes(task_type=2)
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
    paramList, obj = infoin.hyper_parameters(model_id, task_type=2, toc=True)
    return render(request,'discovery/model-details.html',{'obj':obj,'paramList':paramList})

@xframe_options_exempt
def model_training(request):
    return render(request,'discovery/model-training-log-list.html')

@xframe_options_exempt
def getModelLogList(request):
    type_select = request.GET.get('type_select')
    dataset_select = request.GET.get("dataset_select")
    model_select = request.GET.get("model_select")
    page = request.GET.get('page')
    limit = request.GET.get("limit")
    modelList = infoin.model_tdes(task_type=2)
    logList = infoin.run_log()
    if dataset_select == None:
        dataset_select = ''
    if model_select == None:
        model_select = ''
    logList = logList.loc[ logList["model_name"].isin(modelList["model_name"]) ]
    logList = logList.loc[logList["dataset_name"].str.contains(dataset_select) & logList["model_name"].str.contains(model_select)]
    logList = infoin.convert_df_to_list(logList)
    if type_select != '5':
        logList = logList[logList["type"]==type_select]
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
    paramList = infoin.get_run_log_hyper_parameters()
    paramList = paramList.loc[paramList["log_id"]==int(log_id)]
    paramList = infoin.convert_df_to_classlist(paramList)
    ex_data = infoin.run_log()
    ex_data = ex_data.loc[ex_data["log_id"]==int(log_id)]
    ex_data = infoin.convert_df_to_classlist(ex_data)
    
    return render(request,'discovery/model-training-log-parameter.html',{'model_id': log_id,'paramList':paramList,'ex_data':ex_data})

@xframe_options_exempt
def toRunModel(request):
    
    modelList = infoin.model_tdes(task_type=2)
    modelList = infoin.convert_df_to_classlist(modelList)
    datasetList = infoin.dataset()
    datasetList = infoin.convert_df_to_classlist(datasetList)

    result = {}
    result['modelList'] = modelList
    result['datasetList'] = datasetList
    
    return render(request,'discovery/model-training-log-torun.html', result)

@csrf_exempt
def getParamListByModelId(request):
    model_id_select = request.GET.get('model_select')

    if model_id_select == None:
        return JsonResponse({'code':0,'msg':'','count':0,'data':[]})
    
    resultList, _ = infoin.hyper_parameters(int(model_id_select), task_type=2)
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

    modelList = infoin.model_tdes(task_type=2)
    modelName = modelList.loc[modelList["model_id"]==int(model_id)]["model_name"].values[0]
    if modelName in ['ODC', 'SelfORE']:
        Known_Ratio = 0.0
    args = {
        "dname": dataset_name_select,
        "method": modelName,
        "task_type":'relation_discover',
        "kcr":Known_Ratio,
        "lab_ratio":Annotated_Ratio,
        "seed": 0,
        "istrain": 1
    }
    
    name = "{}_{}_{}_{}_{}.pkl".format(dataset_name_select.lower(), 0, "normal", Known_Ratio, float(Annotated_Ratio))
    mid_dir = os.path.join(infoin.result_path, "relation_discover", modelName)
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
        # print(run_log)
        run_log.loc[run_log["log_id"]==log_id, "type"] = type_after
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



@xframe_options_exempt
def model_test(request):
    datasetList = infoin.dataset()
    modelList_detection = infoin.model_tdes(task_type=2)
    df = infoin.run_log()
    if request.GET.get('log_id'):  
        log_id = request.GET.get('log_id')
        cond = (df["log_id"]==int(log_id) )
        cond = cond & (df["type"]==2)
        create_time_new = df.loc[cond].tail(1)
    else:
        create_time_new = df.loc[(df["type"]==2) & (df["model_name"].isin(modelList_detection["model_name"].values.tolist()))]
        if len(create_time_new) > 0:
            create_time_new = create_time_new.iloc[-1, :]
            log_id = create_time_new['log_id']
        else:
            log_id = -1
    if int(log_id) > -1:
        if not isinstance(log_id, str):
            log_id = int(log_id)
        dataset_new = create_time_new['dataset_name']
        if not isinstance(dataset_new, str):
            dataset_new = dataset_new.values[0]
        model_new = create_time_new['model_name']
        if not isinstance(model_new, str):
            model_new = model_new.values[0]
        create_time = df.loc[(df['dataset_name']==dataset_new) & (df['model_name']==model_new) & (df['type']==2)]
        parameters = infoin.get_run_log_hyper_parameters()
        parameters = parameters.loc[parameters["log_id"]==int(log_id)]
        c_time = create_time_new['create_time']
        if not isinstance(c_time, str):
            c_time = c_time.values[0]
    else:
        create_time_new = []
        dataset_new = "No method finised!"
        model_new = "No method finised!"
        create_time = []
        c_time = ""
    
    tranf = infoin.convert_df_to_classlist
    result = {}
    result['datasetList'] = tranf(datasetList)
    result['modelList_detection'] = tranf(modelList_detection)
    result['create_time'] = tranf(create_time)
    result['create_time_new'] = tranf(create_time_new)
    result['parameters'] = tranf(parameters)
    result['show_selected'] = toclass({'dataname': dataset_new, 'modelname': model_new, \
        'createdtime': c_time, 'log_id': int(log_id)})
    
    return render(request,'discovery/model-test.html' , result)
#--------------------------------------------------------------------------------------------------

@csrf_exempt
def show_test_result(request):
    dataset_name = request.GET.get('dataset_name')
    method = request.GET.get('method')
    log_id = request.GET.get('log_id')#new
    return_list = []

    run_log = infoin.run_log()
    run_log_path = run_log.loc[run_log["log_id"]==int(log_id)]["Local_Path"].values[0]
    if os.path.isfile(run_log_path):
        data = load_pickle(run_log_path)
    else:
        return JsonResponse({'code':200, 'msg':'There is no data', 'count':0, 'data':list([]) })
    return_list = data["results"]
    return_list['B3'] = round(return_list['B3']['F1']*100, 2)
    results = {}
    results['data'] = return_list

    return JsonResponse({'code':200,'msg':'Successfully !','data':results})

@csrf_exempt
def model_evaluation_getDataOfTFFineByKey(request):
    key = request.GET.get('key')
    split_key = key.split('_')
    log_id = split_key[-1]
    res = infoin.get_true_flase(log_id, fine=True, tt=2)
    if res is None:
        return JsonResponse({ "code":201, "msg": "There is no data" })
    data_iokir = res
    
    results = {}
    results['code'] = 200
    results['msg'] = ''
    results['data'] = data_iokir
    return JsonResponse(results)

#--------------------------------------------------------------------------------------------------
@xframe_options_exempt
def model_analysis(request):
    dataset_list = infoin.dataset()
    modelList_discovery = infoin.model_tdes(task_type=2)
    run_log = infoin.run_log()
    if request.GET.get('log_id'):
        log_id = request.GET.get('log_id')
        run_log_select= run_log.loc[(run_log["log_id"]==int(log_id))&(run_log["type"]==2)]
        run_log_select = run_log_select.loc[run_log_select["model_name"].isin(modelList_discovery["model_name"])]
        create_time_new = run_log_select.tail(1)
    else:
        run_log_select = run_log.loc[run_log["type"]==2]
        run_log_select = run_log_select.loc[run_log_select["model_name"].isin(modelList_discovery["model_name"])]
        create_time_new = run_log_select.tail(1)
        if len(create_time_new) > 0:
            log_id = create_time_new['log_id'].values[0]
        else:
            log_id = -1
    if log_id >-1:
        example_list = infoin.Model_Test_Example(log_id)
        dataset_new = create_time_new['dataset_name'].values[0]
        model_new = create_time_new['model_name'].values[0]
        create_time = run_log.loc[(run_log["dataset_name"]==dataset_new) & (run_log["model_name"]==model_new) & (run_log["type"]==2)]
        c_time = create_time_new['create_time']
        if not isinstance(c_time, str):
            c_time = c_time.values[0]
    else:
        example_list = []
        create_time_new = []
        dataset_new = "No method finised!"
        model_new = "No method finised!"
        create_time = []
        c_time = ""
    trans = infoin.convert_df_to_classlist
    result = {}
    result['dataset_list'] = trans(dataset_list)
    result['modelList_discovery'] = trans(modelList_discovery)
    result['exampleList'] = trans(example_list)
    result['create_time'] = trans(create_time)
    result['create_time_new'] = trans(create_time_new)
    result['show_selected'] = toclass({'dataname': dataset_new, 'modelname': model_new, \
        'createdtime': c_time, 'log_id': int(log_id)})
    return render(request,'discovery/model-analysis.html' , result)

@csrf_exempt
def model_evaluation_getDataOfTFOverallByKey(request):
    key = request.GET.get('key')
    split_key = key.split('_')
    log_id = split_key[-1]
    res = infoin.get_true_flase(log_id, tt=2)
    if res is None:
        return JsonResponse({ "code":201, "msg": "There is no data" })
    results = {}
    results['code'] = 200
    results['msg'] = ''
    results['data'] = res
    return JsonResponse(results)

@csrf_exempt
def model_analysis_getClassListByDatasetNameAndMethod(request):
    dataset_name = request.GET.get('dataset_name')
    method = request.GET.get('method')
    log_id = str(request.GET.get('log_id'))
    class_type = 'open'
    page = request.GET.get('page')
    limit = request.GET.get("limit")
    return_list = infoin.get_test_example(log_id, dataset_name, method, class_type=class_type)
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
    log_id = str(request.GET.get('log_id'))
    method = request.GET.get('method')
    label = request.GET.get('label_name')
    class_type = 'open'
    page = request.GET.get('page')
    limit = request.GET.get("limit")
    res = infoin.get_test_example(log_id, dataset_name, method, label, class_type=class_type)
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
def model_analysis_getDataOfDINByKey(request):
    key = request.GET.get('key')
    split_key = key.split('_')
    log_id = split_key[-1]
    res = infoin.get_scatter_data(log_id, task_type=2)
    if res is None:
        return JsonResponse({ "code":201, "msg": "There is no data" })
    data_iokir = res
    
    results = {}
    results['code'] = 200
    results['msg'] = ''
    results['data'] = data_iokir
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





