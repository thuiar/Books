from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.clickjacking import xframe_options_exempt
from django.views.decorators.csrf import csrf_exempt
from django.core.paginator import Paginator
from info.info_interaction import infoin, toclass

# Create your views here.
@xframe_options_exempt
def data_annotation(request):#
    
    dataset_name = request.GET.get('dataset_name')
    model_detection = request.GET.get("model_detection")

    dataset_List = infoin.dataset()
    modelList_detection = infoin.model_tdes(task_type=1)
    modelList_discovery = infoin.model_tdes(task_type=2)

    run_log = infoin.run_log()
    run_log_select = run_log.loc[run_log["type"]==2]
    discovery_run_log = run_log_select.loc[run_log_select["model_name"].isin(modelList_discovery["model_name"])]
    detection_run_log = run_log_select.loc[run_log_select["model_name"].isin(modelList_detection["model_name"])]

    discovery_run_log = discovery_run_log.loc[discovery_run_log["dataset_name"].isin(detection_run_log["dataset_name"])]
    detection_run_log = detection_run_log.loc[detection_run_log["dataset_name"].isin(discovery_run_log["dataset_name"])]
    
    dataset_name = "Wiki80"
    if len(detection_run_log) < 1:
        detection_name = "No method finised!"
    else:
        detection_run_log = detection_run_log.tail(1)
        detection_name = detection_run_log["model_name"].values[0]
        dataset_name = detection_run_log["dataset_name"].values[0]
    
    if len(discovery_run_log) < 1:
        discovery_name = "No method finised!"
    else:
        discovery_run_log = (discovery_run_log.loc[discovery_run_log["dataset_name"] == dataset_name]).tail(1)
        discovery_name = discovery_run_log["model_name"].values[0]

    transf = infoin.convert_df_to_classlist
    result = {}
    result['dataset_name'] = dataset_name
    result['dataset_list'] = transf(dataset_List)
    result['model_detection'] = model_detection
    result['modelList_detection'] = transf(modelList_detection)
    result['modelList_discovery'] = transf(modelList_discovery)
    result['show_selected'] = toclass({
        'dataname': dataset_name, 
        'detect_modelname': detection_name, 
        'discovery_modelname': discovery_name
    })

    return render(request,'annotation/data_annotation.html',result)

@csrf_exempt
def getDatasetList(request):
    detection_method = request.GET.get('detection_method')
    discovery_method = request.GET.get('discovery_method')
    know_relation = request.GET.get('konw_relation')
    labeled_ratio = request.GET.get('labeled_ratio')
    dataset_list = request.GET.get('dataset_list')
    dataset_list = infoin.get_pip_datalist(
        dataset_list, detection_method, discovery_method,
        know_relation, labeled_ratio
    )
    if dataset_list is None:
        dataset_list = []
    count = len(dataset_list)

    dataset_list_arr=[]
    dataset_list_arr.append(dataset_list)

    result = {}
    result['code'] = 0
    result['msg'] = ''
    result['count'] = count
    result['data'] = dataset_list_arr
    
    return JsonResponse(result)

@csrf_exempt
def getClassListByDatasetNameAndClassType(request):
    dataset_name = request.GET.get('dataset_name')
    class_type = request.GET.get('class_type')
    page = request.GET.get('page')
    limit = request.GET.get("limit")
    detection_method = request.GET.get('detection_method')
    discovery_method = request.GET.get('discovery_method')
    know_relation = request.GET.get('konw_relation')
    labeled_ratio = request.GET.get('labeled_ratio')
    dataset_list = request.GET.get('dataset_list')

    return_list = infoin.get_pipclasslist_by_key(
        dataset_name, detection_method, discovery_method, 
        know_relation, labeled_ratio, class_type=class_type
    )
    count = len(return_list)
    paginator = Paginator(return_list, limit)
    return_list = paginator.get_page(page)

    results = {}
    results['code'] = 0
    results['msg'] = ''
    results['count'] = count
    results['data'] = list(return_list)
    return JsonResponse(results)

def getTextListByDatasetClassTypeLabelName(request):
    dataset_name = request.GET.get('dataset_name')
    class_type = request.GET.get('class_type')
    label_name = request.GET.get('label_name')
    page = request.GET.get('page')
    limit = request.GET.get("limit")
    detection_method = request.GET.get('detection_method')
    discovery_method = request.GET.get('discovery_method')
    know_relation = request.GET.get('konw_relation')
    labeled_ratio = request.GET.get('labeled_ratio')
    dataset_list = request.GET.get('dataset_list')
    return_list = infoin.get_pipclasslist_by_key(
        dataset_name, detection_method, discovery_method, 
        know_relation, labeled_ratio, class_type=class_type,
        this_label=label_name
    )
    
    count = len(return_list)
    paginator = Paginator(return_list, limit)
    return_list = paginator.get_page(page)

    results = {}
    results['code'] = 0
    results['msg'] = ''
    results['count'] = count
    results['data'] = list(return_list)
    return JsonResponse(results)

def getTextListByDatasetForUnknown(request):
    dataset_name = request.GET.get('dataset_name')
    class_type = 'open' # all open === unknown
    page = request.GET.get('page')
    limit = request.GET.get("limit")
    detection_method = request.GET.get('detection_method')
    discovery_method = request.GET.get('discovery_method')
    know_relation = request.GET.get('konw_relation')
    labeled_ratio = request.GET.get('labeled_ratio')
    dataset_list = request.GET.get('dataset_list')

    result_list = infoin.get_pipclasslist_by_key(
        dataset_name, detection_method, discovery_method, 
        know_relation, labeled_ratio, class_type=class_type,
        this_label="unknown"
    )
    count = len(result_list)
    paginator = Paginator(result_list, limit)
    result_list = paginator.get_page(page)

    results = {}
    results['code'] = 0
    results['msg'] = ''
    results['count'] = count
    results['data'] = list(result_list)
    return JsonResponse(results)
