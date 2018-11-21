from django.urls import path
from SIW import views

urlpatterns = [
    path('login/', views.login, name='login'),
    path('index/', views.index, name='index'),
    path('upload/', views.upload, name='upload'),
    path('creat/(?P<select_list>)', views.creat, name='creat'),
    path('save/', views.save, name='save'),
    path('clear/', views.clear, name='clear'),
    path('scriptmanage/', views.scriptmanage, name='scriptmanage'),
    path('permissionmanage/', views.permissionmanage, name='permissionmanage'),
    path('rolemanage/', views.rolemanage, name='rolemanage'),
    path('usermanage/', views.usermanage, name='usermanage'),
]

handler404 = views.page_not_found
handler500 = views.page_inter_error
