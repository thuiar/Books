# return data
from logging import info
from django.shortcuts import render
# download file
from django.http import FileResponse
# return html
from django.views.decorators.clickjacking import xframe_options_exempt
# return Json
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
# save and read file
import os, sys
# judge platform
import platform
# time
from django.utils import timezone
# page helper
from django.core.paginator import Paginator
from info.info_interaction import infoin

# Create your views here.
@xframe_options_exempt
def toDatasetList(request):
    return render(request,'thedataset/thedataset-list.html')

def update_source(request):
    result = {}
    a = infoin.model_tdes(task_type=1)
    a = infoin.convert_df_to_list(a)
    return JsonResponse(list(a))
@csrf_exempt
def getDatasetList(request):
    type_select = '5'
    dataset_name_select = request.GET.get("dataset_name_select")
    domain_select = request.GET.get("domain_select")
    page = request.GET.get('page')
    limit = request.GET.get("limit")
    dataset_id={1,4,5,}

    if dataset_name_select == None:
        dataset_name_select = ''
    if domain_select == None:
        domain_select = ''
    datasetList = infoin.dataset()
    datasetList = infoin.convert_df_to_list(datasetList)
    count = len(datasetList)
    # 分页
    paginator = Paginator(datasetList, limit)
    datasetList = paginator.get_page(page)
    result = {}
    result['code'] = 0
    result['msg'] = ''
    result['count'] = count
    result['data'] = list(datasetList)
    
    return JsonResponse(result)


@xframe_options_exempt
def toAddHtml(request):
    return render(request,'thedataset/thedataset-add.html')

@csrf_exempt
def addDataset(request):
    ##get base msg
    dataset_name = request.POST['dataset_name']
    domain = request.POST['domain']
    class_num = request.POST['class_num']
    source = request.POST['source']
    sample_total_num = request.POST['sample_total_num']
    sample_training_num = request.POST['sample_training_num']
    sample_validation_num = request.POST['sample_validation_num']
    sample_test_num = request.POST['sample_test_num']
    sentence_max_length = request.POST['sentence_max_length']
    sentence_avg_length = request.POST['sentence_avg_length']
    types = 1    # upload type == 1(User)
    ## get files
    # file_all = request.FILES.get('file_all')
    file_name_list = ['file_train', 'file_dev', 'file_test']
    file_list_dic = {}
    for file_name in file_name_list:
        file_list_dic[file_name] = request.FILES.get(file_name)
        if file_list_dic[file_name] == None:
            return JsonResponse({'code': 201, 'msg': 'Please Chose file:'+ file_name.split('_')[1] +'.tsv !!!'})

    local_path = os.path.join(infoin.data_path, dataset_name.lower())
    
    ## determine if the dataset exists
    if os.path.exists(local_path):
        return JsonResponse({'code':201,'msg':'The Dataset "'+ dataset_name +'" Already Exists , Please Check It !!!'})
    try:
        os.makedirs(local_path)

        ## save files
        for file_name in file_name_list:
            destination = open(local_path + file_name.split('_')[1] +'.json', 'wb+')
            for chunk in file_list_dic[file_name].chunks():
                destination.write(chunk)
            destination.close()
        datalist = infoin.dataset()
        datalist = infoin.dataset(
            dataset_id = len(datalist) + 1,
            dataset_name = dataset_name,
            domain= domain,
            class_num= class_num,
            source= source,
            local_path= local_path,
            type= types,
            sample_total_num=sample_total_num,
            sample_training_num= sample_training_num,
            sample_validation_num= sample_validation_num,
            sample_test_num=sample_test_num,
            sentence_max_length=sentence_max_length,
            sentence_avg_length=sentence_avg_length,
            create_time=timezone.now().strftime("%Y-%m-%d %H:%M:%S")
        )

    except :
        if os.path.exists(local_path):
            for i_path in os.listdir(local_path):
                os.removedirs(i_path)
            os.removedirs(local_path)
        return JsonResponse({'code': 400, 'msg': 'Add Dataset Has An Error ！！'})

    ## return msg
    return JsonResponse({'code':200,'msg':'Successfully Add Dataset'})

@xframe_options_exempt
def details(request):
    dataset_id = request.GET.get('dataset_id')
    obj = infoin.dataset()
    obj = obj.loc[obj["dataset_id"]==int(dataset_id)]
    obj = infoin.convert_df_to_classlist(obj)
    return render(request,'thedataset/thedataset-details.html',{'obj':obj[0]})

@csrf_exempt
def delData(request):
    ##get base msg
    print('there is deldate1')
    dataset_name = request.POST['dataset_name']
    dataset_id = request.POST['dataset_id']
    
    datalist = infoin.dataset()
    record = infoin._cond_select(datalist, dataset_id=dataset_id)
    ts = record['type']
    if isinstance(ts, int):
        ts = int(ts)
    if ts != 1 :
        return JsonResponse({'code': 400, 'msg': 'Cannot Delete Internal Dataset ！！'})
    print(dataset_name)
    
    local_path = os.path.join(infoin.data_path, dataset_name.lower())
    print('*'*20, '\n'*4, 'Delete Begin\t\t', local_path)
    try:
        ## delete data in disk
        if os.path.exists(local_path):
            print('*'*20, '\n'*4, 'Delete Begin\t\t', local_path)
            for i_path in os.listdir(local_path):
                print('*'*20, '\n'*4, 'Delete runing\t\t', os.path.join(local_path, i_path))
                os.remove(os.path.join(local_path, i_path))
                # os.removedirs(os.path.join(local_path, i_path))
            os.removedirs(local_path)
        
        ## delete data in database
        infoin.del_dataset(dataset_name)
    except:
        return JsonResponse({'code': 400, 'msg': 'Delete Dataset Has An Error ！！'})
    return JsonResponse({'code':200,'msg':'Successfully Delete Dataset'})









