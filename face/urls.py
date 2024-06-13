from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [path('',views.main, name='main_page'),
               path('choose_page',views.choose, name='choose_page'),
               path('choose_gender_page',views.choose_gender, name='choose_gender_page'),
               path('upload_page1',views.upload_page1, name='upload_page1'),
               path('upload_page2',views.upload_page2, name='upload_page2'),
               path('upload_page3',views.upload_page3, name='upload_page3'),
               path('upload_page4',views.upload_page4, name='upload_page4'),
               path('result_page',views.result, name='result_page'),
               path('loading_page',views.loading, name='loading_page')]

urlpatterns += static(settings.MEDIA_URL, documnet_root=settings.MEDIA_ROOT)
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_URL)
