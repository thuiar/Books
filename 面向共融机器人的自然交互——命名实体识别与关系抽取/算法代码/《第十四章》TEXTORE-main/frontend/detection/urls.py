"""the dataset  URL Configuration

"""
from django.conf.urls import url
from django.contrib import admin
from django.urls import path
from . import views
urlpatterns = [
    path('model_management/',views.model_management),
    path('model_management/getModelList',views.getModelList),
    url(r'^model_management/details/$',views.model_management_details),

    
    path('model_training/',views.model_training),
    path('model_training/log_delete',views.log_delete),
    path('model_training/getModelList',views.getModelLogList),
    url(r'^model_training/toLogParameter/$',views.toLogParameter),

    path('model_training/toRunModel',views.toRunModel),
    path('model_training/getParamListByModelId',views.getParamListByModelId),
    path('model_training/add_model_training_log',views.add_model_training_log),
    path('model_training/kill_running',views.kill_running),
    
    path('model_test/',views.model_test),
    path('test',views.test),
    
    path('model_test/check_evaluation',views.check_evaluation),
    path('model_test/show_create_time',views.show_create_time),
    path('model_test/show_hyper_parameters',views.show_hyper_parameters),
    path('model_test/show_test_result',views.show_test_result),
    path('model_test/model_evaluation_getDataOfIOLRByKey',views.model_evaluation_getDataOfIOLRByKey),
    path('model_test/model_evaluation_getDataOfTFOverallByKey',views.model_evaluation_getDataOfTFOverallByKey),
    path('model_test/model_evaluation_getDataOfTFFineByKey',views.model_evaluation_getDataOfTFFineByKey),
    
    path('model_analysis/',views.model_analysis),
    path('model_analysis/modelAnalysisTest',views.modelAnalysisTest),
    path('model_analysis/model_analysis_getClassListByDatasetNameAndMethod',views.model_analysis_getClassListByDatasetNameAndMethod),
    path('model_analysis/model_analysis_getTextListByDatasetNameAndMethodAndLabel',views.model_analysis_getTextListByDatasetNameAndMethodAndLabel),

    path('model_analysis/model_analysis_getDataByKey',views.model_analysis_getDataByKey),
]
