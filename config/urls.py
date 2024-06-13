from django.contrib import admin
from django.urls import path, include
from face import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('face/', include('face.urls')),
]
