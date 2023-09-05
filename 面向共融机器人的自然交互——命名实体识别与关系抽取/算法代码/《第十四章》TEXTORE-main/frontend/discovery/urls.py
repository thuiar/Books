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

    
    path('model_test/',views.model_test),
    path('model_test/show_test_result',views.show_test_result),
    path('model_test/model_evaluation_getDataOfTFOverallByKey',views.model_evaluation_getDataOfTFOverallByKey),
    path('model_test/model_evaluation_getDataOfTFFineByKey',views.model_evaluation_getDataOfTFFineByKey),
    
    path('model_analysis/',views.model_analysis),
    path('model_analysis/model_analysis_getClassListByDatasetNameAndMethod',views.model_analysis_getClassListByDatasetNameAndMethod),
    path('model_analysis/model_analysis_getTextListByDatasetNameAndMethodAndLabel',views.model_analysis_getTextListByDatasetNameAndMethodAndLabel),
    path('model_analysis/model_analysis_getDataOfDINByKey',views.model_analysis_getDataOfDINByKey)
]

























