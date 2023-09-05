# return data
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
# database tables
from thedataset import models
# page helper
from django.core.paginator import Paginator
# serializers

sys.path.append("..") 
        
backend_engine_linux = '/../open_intent_detection/run.py '
backend_engine_win = '\\..\\textoir\\run_detect.py '

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
    
    modelList = models.Model_Tdes.objects.values().filter(model_name__contains=model_name_select,type=1).order_by('model_id')
    count = modelList.count()
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
    obj = models.Model_Tdes.objects.get(model_id=model_id)
    paramList = models.Hyper_parameters.objects.filter(model_id=model_id)
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

    
    logList = models.Run_Log.objects.values().filter(model_id__type=1)

    if dataset_select == None:
        dataset_select = ''
    if model_select == None:
        model_select = ''
    logList = logList.filter(dataset_name__contains=dataset_select,model_name__contains=model_select).order_by('-log_id')
    if type_select != '5':
        logList = logList.filter(type=type_select)

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
    paramList = models.run_log_hyper_parameters.objects.filter(log_id=log_id)
    ex_data = models.Run_Log.objects.values().filter(log_id=log_id)
    return render(request,'detection/model-training-log-parameter.html',{'model_id': log_id,'paramList':paramList,'ex_data':ex_data})

@xframe_options_exempt
def toRunModel(request):
    
    # get model list
    modelList = models.Model_Tdes.objects.values().filter(type=1)
    datasetList = models.DataSet.objects.values()

    result = {}
    result['modelList'] = modelList
    result['datasetList'] = datasetList
    
    return render(request,'detection/model-training-log-torun.html', result)

@csrf_exempt
def getParamListByModelId(request):
    model_id_select = request.GET.get('model_select')

    if model_id_select == None:
        return JsonResponse({'code':0,'msg':'','count':0,'data':[]})
    
    resultList = models.Hyper_parameters.objects.values().filter(model_id=model_id_select).order_by('param_id')
    
    paginator = Paginator(resultList, 100)
    resultList = paginator.get_page(1)

    result = {}
    result['code'] = 0
    result['msg'] = ''
    result['count'] = paginator.count
    result['data'] = list(resultList)
    return JsonResponse(result)

def json_read(path):
    
    with open(path, 'r')  as f:
        json_r = json.load(f)

    return json_r

def json_add(predict_t_f, path):
    
    with open(path, 'w') as f:
        json.dump(predict_t_f, f, indent=4)

def get_key_attrs(path):
    string=''
    with open(path, 'r') as lines:

        flag = False
        for line in lines:

            if line.strip() == "Args:":
                flag = True
                continue
            
            if line.startswith("hyper_parameters"):
                flag = False
                break

            if flag:
                string += line
    
    string = string.replace("\"","")
    sen_list = string.split('\n')
    key_attr_dict = {}

    for sen in sen_list:
        if ':' not in sen:
            break

        sen_split = sen.split(':')
        key_attr_split = sen_split[0].strip().split('(')
        attr = key_attr_split[1].strip(')')
        key = key_attr_split[0].strip()
        
        key_attr_dict[key] = attr

    return key_attr_dict

def convert_type(attr, value):

    if attr == 'int':
        value = int(value)

    elif attr == 'str':
        value = str(value)

    elif attr == 'binary':
        if value == 'True':
            value = True
        elif value == 'False':
            value = False

    elif attr == 'autofill':
        value = None
    
    elif attr == 'float':
        value = float(value)
    
    elif attr == 'directory':
        value = str(value)

    return value

def get_config_py(method_name, method_type):

    if method_type == 1:
        rootpath = os.path.abspath(os.path.join(os.getcwd(), ".."))+'/open_intent_detection/configs/'
    elif method_type == 2: 
        rootpath = os.path.abspath(os.path.join(os.getcwd(), ".."))+'/open_intent_discovery/configs/'
   
    save_file_name = method_name+'.py'
    results_path = os.path.join(rootpath, save_file_name)
    
    key_attr_dict = get_key_attrs(results_path)

    string=''
    op=False
    with open(results_path, 'r') as lines:
        for line in lines:
            if line.strip().startswith("hyper_parameters"):
                op=True
                continue
            if line.strip()=="}":
                op=False
                break
            if op:
                string = string + line.strip()
    

    string = string.replace("\"","")
    string = string.replace("\'","")
    string = string.replace(" ", "")
    hyper_parameters = string.split(',')

    hyper_parameter_dict = {}
    import copy
    tmp_hyper_parameters = copy.copy(hyper_parameters)

    for item in tmp_hyper_parameters:
        if ':' not in item:
            continue
        
        key_value = item.split(':')

        key = key_value[0].strip()
        value = key_value[1].strip()

        hyper_parameter_dict[key] = value

        attr = key_attr_dict[key]
        hyper_parameter_dict[key] = convert_type(attr, value)

    return hyper_parameter_dict, key_attr_dict

@csrf_exempt
def add_model_training_log(request):   

    print('there is add_model_training_log')
    model_id = request.POST['model_id']
    dataset_name_select = request.POST['dataset_name_select']
    Known_Intent_Ratio = request.POST['Known_Intent_Ratio']
    Annotated_Ratio = request.POST['Annotated_Ratio']
    params = request.POST['params']
    paramsListJson = json.loads(params)

    modelItem = models.Model_Tdes.objects.get(model_id=model_id)
    model_name = modelItem.model_name  

    save_file_name = 'config.json'
    if modelItem.type == 1:
        save_dir = sys.path[0]+'/static/jsons/open_intent_detection/'
    elif modelItem.type == 2:
        save_dir = sys.path[0]+'/static/jsons/open_intent_discovery/'
    else:
        print('This type is not implemented.')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    results_path = os.path.join(save_dir, save_file_name)    

    if not os.path.exists(results_path):
        f = open(results_path, 'w')

    json_data={}
    
    default_parameters, attrs  = get_config_py(model_name, modelItem.type)

    for key in default_parameters.keys():
        json_data[key] = default_parameters[key]

    for paramItem in paramsListJson:

        attr = attrs[paramItem['param_name']]
        json_data[paramItem['param_name']] = convert_type(attr, paramItem['default_value'])

    json_add(json_data, results_path)

    para_str_python = ' --dataset '+  dataset_name_select + ' --known_cls_ratio ' + Known_Intent_Ratio + ' --labeled_ratio ' + Annotated_Ratio + ' --method ' + model_name + ' --config_file_name '+model_name+'.py '+  ' --save_model'+ ' --train '+' --save_frontend_results '   

    try:
        run_logItem = models.Run_Log( 

            dataset_name = dataset_name_select, 
            model_name = modelItem.model_name,
            model_id_id = model_id,
            Local_Path = "",
            create_time = timezone.now().strftime("%Y-%m-%d %H:%M:%S"),
            Annotated_ratio = Annotated_Ratio, 
            Intent_Ratio = Known_Intent_Ratio,
            type = 1 # runing state
            
            )

        run_logItem.save()
       
        if platform.system() == 'Linux':
            local_path = sys.path[0]+'/static/log/detection/add_model_training_log/running/'+  dataset_name_select + model_id + Annotated_Ratio + Known_Intent_Ratio + str(run_logItem.log_id) + '/'
        elif platform.system() == 'Windows':
            local_path = sys.path[0]+'\\static\\log\\detection\\add_model_training_log\\running\\'+  dataset_name_select + model_id + Annotated_Ratio + Known_Intent_Ratio + str(run_logItem.log_id) + '\\'
        
        _ = models.Run_Log.objects.filter(log_id=run_logItem.log_id).update(Local_Path = local_path)

        # save msg to run_log_hyper_parameters
        for paramItem in paramsListJson:
            run_log_hyper_parametersItem = models.run_log_hyper_parameters(
                    param_name = paramItem['param_name'],param_describe = paramItem['param_describe'],
                    value_type = paramItem['value_type'],default_value = paramItem['default_value'],
                    run_value = paramItem['default_value'],log_id = run_logItem.log_id
                )
            run_log_hyper_parametersItem.save()

        os.makedirs(local_path)

        str_run = ''
        if platform.system() == 'Linux':
           
            str_run = 'python ' + str(sys.path[0])+ str(backend_engine_linux) +  str(para_str_python) +' --log_id '+str(run_logItem.log_id) #最终执行跑模型命令语句————-——————————————————————————————————————

        elif platform.system() == 'Windows':
            str_run = 'python ' + sys.path[0]+ backend_engine_win + para_str_python +' --log_id '+str(run_logItem.log_id)
        

        str_run = shlex.split(str_run)
        process = subprocess.Popen(str_run)
        # #get pid
        run_pid = process.pid
        # run_pid = 99999   
        ## save msg to run log
        updat = models.Run_Log.objects.filter(log_id=run_logItem.log_id).update(run_pid = run_pid)
        
        ##get returncode ---wait runing state change
        process.communicate()
        run_type = process.returncode
        # ## save run_type to run log
        # 1--runing  2--finished 3--failed
        if run_type == 0 or run_type == '0':
            type_after = 2
        else :
            type_after = 3
        # update runing state to run_log
        updat = models.Run_Log.objects.filter(log_id=run_logItem.log_id).update(run_pid = run_pid,type=type_after)

    except :
        updat = models.Run_Log.objects.filter(log_id=run_logItem.log_id).update(type=3)
        return JsonResponse({'code': 400, 'msg': 'Run  Process Has An Error ！！'})

    finally:
        if os.path.exists(local_path):
            os.removedirs(local_path)

    print("Running finished!")

    return JsonResponse({'code':200,'msg':'Successfully Running Process!'})


#kill_run
#need pid and stop
@csrf_exempt
def kill_running(request):
    try:
        run_pid = request.POST.get('run_pid')
        log_id = request.POST.get('log_id')
        if models.Run_Log.objects.get(log_id=log_id).type != 1:
            return JsonResponse({'code': 400, 'msg': 'Process '+run_pid+' Was Over ！！'})
        command = 'kill -9 ' + str(run_pid)
        command = shlex.split(command)
        subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        _ = models.Run_Log.objects.filter(log_id=log_id).update(type=3)
        os.removedirs(models.Run_Log.objects.get(log_id=log_id).Local_Path)

    except:
        return JsonResponse({'code': 400, 'msg': 'Kill  Process Has An Error ！！'})
    return JsonResponse({'code':200,'msg':'Successfully Kill Process!'})

@xframe_options_exempt
def model_test(request):

    datasetList = models.DataSet.objects.values()
    modelList_detection = models.Model_Tdes.objects.values().filter(type = 1)

    if request.GET.get('log_id'):  
        log_id = request.GET.get('log_id')
        create_time_new = models.Run_Log.objects.values().filter(log_id = log_id,model_id__type=1,type = 2).first()#,type = 2)     #默认显示最新的一条
    else:
        create_time_new = models.Run_Log.objects.values().filter(model_id__type=1,type = 2).order_by('-log_id').first()#,type = 2)     #默认显示最新的一条
        log_id = create_time_new['log_id']
    
    dataset_new = create_time_new['dataset_name']
    model_new = create_time_new['model_name']

    create_time = models.Run_Log.objects.values().filter(dataset_name = dataset_new,model_name=model_new,model_id__type=1,type = 2)#,type = 2) #成功完成的记录 *******待改***********
 
    parameters = models.run_log_hyper_parameters.objects.values().filter(log_id=log_id)
    
    result = {}
    result['datasetList'] = datasetList
    result['modelList_detection'] = modelList_detection
    result['create_time'] = create_time
    result['create_time_new'] = create_time_new
    result['parameters'] = parameters
    
    return render(request,'detection/model-test.html' , result)


@csrf_exempt 
def check_evaluation(request):
    log_id= request.GET.get('log_id')
    
    
    modelList = models.Run_Log.objects.values().filter(log_id=log_id)
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
        model_name_default = "Open Intent Detection"
    model_name_default = model_name
    
    create_time = models.Run_Log.objects.values().filter(dataset_name=dataset_name,model_name=model_name,type = 2).order_by('-log_id')
 
    datasetList = models.DataSet.objects.values()
    modelList_detection = models.Model_Tdes.objects.values().filter(type = 1)
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
    json_path = os.path.join(sys.path[0], 'static/jsons/open_intent_detection/', 'json_test_results.json')
    return_list = []

    with open(json_path, 'r') as load_f:
        load_dict = json.load(load_f)
    if load_dict.__contains__(dataset_name +'_'+ method +"_"+ log_id) == False:
        return JsonResponse({'code':200, 'msg':'There is no data', 'count':0, 'data':list([]) })
    return_list = load_dict[dataset_name +'_'+ method +"_"+ log_id ]
    results = {}
    results['data'] = return_list


    return JsonResponse({'code':200,'msg':'Successfully !','data':results})

def show_hyper_parameters(request):
  
    
    log_id= request.GET.get('log_id')

    parameters = models.run_log_hyper_parameters.objects.values().filter(log_id=log_id)
    result = {}
    ratio=[]
    result = list(parameters)
    ratio.append( models.Run_Log.objects.values().filter(log_id=log_id).first()['Intent_Ratio'])
    ratio.append( models.Run_Log.objects.values().filter(log_id=log_id).first()['Annotated_ratio'])
    
    return JsonResponse({'code':200,'msg':'Successfully !','data':result,'ratio':ratio})
    
@csrf_exempt    
def getModelAnalysisExampleData(request):
    def read(path):
        reader = csv.reader(open(path), delimiter = '\t', quotechar = None)
        lines = []
        for i,j in enumerate(reader):
            if i == 0 :
                continue
            lines.append(j[0]+'\n')
        return lines
    ##get base msg
    example_num = request.POST['example_num']
    str_path = ''
    if example_num == 'Example-1':
        str_path = 'test1_data'
    elif example_num == 'Example-2':
        str_path = 'test2_data'
    elif example_num == 'Example-3':
        str_path = 'test3_data'
    ## genernate local_path
    local_path = ''
    if platform.system() == 'Linux':
        local_path = sys.path[0]+'/static/test_data/test_data/'+  str_path + '/test.tsv'
    elif platform.system() == 'Windows':
        local_path = sys.path[0]+'\\static\\test_data\\test_data\\'+  str_path + '\\test.tsv'
    ##
    text = read(local_path)
    ## return msg
    return JsonResponse({'code':200,'msg':'Successfully Add Dataset','text':text})


@csrf_exempt
def modelAnalysisTest(request):
    model_detection = request.POST['model_detection']
    example_select_detection = request.POST['example_select_detection']

    return JsonResponse({'code':200,'msg':'Successfully Detection The Intent!'})

@xframe_options_exempt
def model_analysis(request):#
    dataset_list = models.DataSet.objects.values()
    modelList_detection = models.Model_Tdes.objects.values().filter(type = 1)
    example_list = models.Model_Test_Example.objects.values()


    if request.GET.get('log_id'):     #如果是跳转，带着log_id_jump
        log_id = request.GET.get('log_id')
        create_time_new = models.Run_Log.objects.values().filter(log_id = log_id,model_id__type=1,type = 2).first()
    else:
        create_time_new = models.Run_Log.objects.values().filter(model_id__type=1,type = 2).order_by('-log_id').first()
        log_id = create_time_new['log_id']

    dataset_new = create_time_new['dataset_name']
    model_new = create_time_new['model_name']
    create_time = models.Run_Log.objects.values().filter(dataset_name = dataset_new,model_name=model_new,model_id__type=1,type = 2) 


    result = {}
    result['dataset_list'] = dataset_list
    result['modelList_detection'] = modelList_detection
    result['exampleList'] = example_list
    result['create_time'] = create_time
    result['create_time_new'] = create_time_new
    
    return render(request,'detection/model-analysis.html',result)



def test(request):
    results={}
    
    models.DataSet.objects.values().delete()
    a='okk'


    return JsonResponse(list(a),safe=False)

@csrf_exempt
def model_evaluation_getDataOfTFOverallByKey(request):
    key = request.GET.get('key')
    json_path = os.path.join(sys.path[0], 'static/jsons/open_intent_detection/', 'true_false_overall.json')
    data_iokir = {}
    with open(json_path, 'r') as load_f:
        load_dict = json.load(load_f)
    if not load_dict.__contains__(key):
        return JsonResponse({ "code":201, "msg": "There is no data" })
    data_iokir = load_dict[key]
    
    results = {}
    results['code'] = 200
    results['msg'] = ''
    results['data'] = data_iokir
    return JsonResponse(results)


@csrf_exempt
def model_evaluation_getDataOfTFFineByKey(request):
    key = request.GET.get('key')
    json_path = os.path.join(sys.path[0], 'static/jsons/open_intent_detection/', 'true_false_fine.json')
    data_iokir = {}
    with open(json_path, 'r') as load_f:
        load_dict = json.load(load_f)
    if not load_dict.__contains__(key):
        return JsonResponse({ "code":201, "msg": "There is no data" })
    data_iokir = load_dict[key]
    
    results = {}
    results['code'] = 200
    results['msg'] = ''
    results['data'] = data_iokir
    return JsonResponse(results)

@csrf_exempt
def model_evaluation_getDataOfTFFineByKey(request):
    key = request.GET.get('key')
    json_path = os.path.join(sys.path[0], 'static/jsons/open_intent_detection/', 'true_false_fine.json')
    data_iokir = {}
    with open(json_path, 'r') as load_f:
        load_dict = json.load(load_f)
    if not load_dict.__contains__(key):
        return JsonResponse({ "code":201, "msg": "There is no data" })
    data_iokir = load_dict[key]
    
    results = {}
    results['code'] = 200
    results['msg'] = ''
    results['data'] = data_iokir
    return JsonResponse(results)


@csrf_exempt
def model_evaluation_getDataOfIOLRByKey(request):
    key = request.GET.get('key')
    json_path = os.path.join(sys.path[0], 'static/jsons/open_intent_detection/', 'json_detection_results.json')
    data_iokir = {}
    with open(json_path, 'r') as load_f:
        load_dict = json.load(load_f)
    if not load_dict.__contains__(key):
        return JsonResponse({ "code":201, "msg": "There is no data" })
    data_iokir = load_dict[key]
    
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
    json_path = os.path.join(sys.path[0], 'static/jsons/open_intent_detection/', 'analysis_table_info.json')
    return_list = []

    with open(json_path, 'r') as load_f:
        load_dict = json.load(load_f)
    if load_dict.__contains__("class_list_"+ dataset_name +'_'+ method +"_"+ log_id +"_" + class_type) == False:
        return JsonResponse({'code':200, 'msg':'There is no data', 'count':0, 'data':list([]) })
    
    return_list = load_dict["class_list_"+ dataset_name +'_'+ method +"_"+ log_id +"_"+ class_type]
    
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
    json_path = os.path.join(sys.path[0], 'static/jsons/open_intent_detection/', 'analysis_table_info.json')
    return_list = []
    with open(json_path, 'r') as load_f:
        load_dict = json.load(load_f)
    key = "text_list_"+ str(dataset_name) +'_'+ str(method) +"_"+ str(log_id) +"_"+ str(class_type)+"_"+str(label)

    if load_dict.__contains__(key) == False:
        return JsonResponse({'code':200, 'msg':'There is no data', 'count':0, 'data':list([]) })
    
    return_list = load_dict[key]
    
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
def model_analysis_getDataOfADBByKey(request):
 
    key = request.GET.get('key')           
    json_path = os.path.join(sys.path[0], 'static/jsons/open_intent_detection/', 'ADB_analysis.json')
    data_ADB = {}
    with open(json_path, 'r') as load_f:
        load_dict = json.load(load_f)
    if not load_dict.__contains__(key):
        return JsonResponse({ "code":201, "msg": "There is no data" })
    data_ADB = load_dict[key]
    
    results = {}
    results['code'] = 200
    results['msg'] = ''
    results['data'] = data_ADB
    return JsonResponse(results)


@csrf_exempt
def model_analysis_getDataOfMSPByKey(request):

    key = request.GET.get('key')           
    json_path = os.path.join(sys.path[0], 'static/jsons/open_intent_detection/', 'MSP_analysis.json')
    data_chart = {}
    with open(json_path, 'r') as load_f:
        load_dict = json.load(load_f)
    if not load_dict.__contains__(key):
        return JsonResponse({ "code":201, "msg": "There is no data" })
    data_chart = load_dict[key]
    
    results = {}
    results['code'] = 200
    results['msg'] = ''
    results['data'] = data_chart
    return JsonResponse(results)


@csrf_exempt
def model_analysis_getDataOfDeepUnkByKey(request):

    key = request.GET.get('key')           
    json_path = os.path.join(sys.path[0], 'static/jsons/open_intent_detection/', 'DeepUnk_analysis.json')
    data_chart = {}
    with open(json_path, 'r') as load_f:
        load_dict = json.load(load_f)
    if not load_dict.__contains__(key):
        return JsonResponse({ "code":201, "msg": "There is no data" })
    data_chart = load_dict[key]
    
    results = {}
    results['code'] = 200
    results['msg'] = ''
    results['data'] = data_chart
    return JsonResponse(results)

@csrf_exempt
def model_analysis_getDataOfDOCByKey(request):

    key = request.GET.get('key')           
    json_path = os.path.join(sys.path[0], 'static/jsons/open_intent_detection/', 'DOC_analysis.json')
    data_chart = {}
    with open(json_path, 'r') as load_f:
        load_dict = json.load(load_f)
    if not load_dict.__contains__(key):
        return JsonResponse({ "code":201, "msg": "There is no data" })
    data_chart = load_dict[key]
    
    results = {}
    results['code'] = 200
    results['msg'] = ''
    results['data'] = data_chart
    return JsonResponse(results)

@csrf_exempt
def model_analysis_getDataOfOpenMaxByKey(request):

    key = request.GET.get('key')           
    json_path = os.path.join(sys.path[0], 'static/jsons/open_intent_detection/', 'OpenMax_analysis.json')
    data_chart = {}
    with open(json_path, 'r') as load_f:
        load_dict = json.load(load_f)
    if not load_dict.__contains__(key):
        return JsonResponse({ "code":201, "msg": "There is no data" })
    data_chart = load_dict[key]
    
    results = {}
    results['code'] = 200
    results['msg'] = ''
    results['data'] = data_chart
    return JsonResponse(results)

def log_delete(request):
    log_id =  request.GET.get('log_id')


    models.Run_Log.objects.filter(log_id=log_id).delete()
    results = {}
    results['code'] = 200
    results['msg'] = 'del_okk'
 
    return JsonResponse(results)





