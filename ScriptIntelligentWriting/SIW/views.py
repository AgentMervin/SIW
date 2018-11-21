from django.shortcuts import render
from SIW.models import TbFile, TbUser
from datetime import datetime


def login(request):
    error_msg = "现 在 登 录"
    if request.method == "POST":
        name = request.POST['Name']
        password = request.POST['Password']
        isLogin = request.POST.getlist('IsLogin')

        if TbUser.objects.filter(user_id=name, user_password=password).count() > 0:
            files = TbFile.objects.all()
            return render(request, "index.html", {'files': files})
        else:
            error_msg = "用 户 密 码 不 匹 配"
    return render(request, "login.html", {'error_msg': error_msg})


def index(request):
    files = TbFile.objects.all()
    return render(request, "index.html", {'files': files})


def upload(request):
    if request.method == "POST":
        uploadfiles = request.FILES.getlist("Files")
        if uploadfiles:
            for file in uploadfiles:
                if not TbFile.objects.filter(file_name=str(file.name).split(".")[0]):
                    destination = open("./Data/" + file.name, "wb+")
                    for chunk in file.chunks():
                        destination.write(chunk)
                    TbFile.objects.create(file_name=str(file.name).split(".")[0],
                                          file_type=str(file.name).split(".")[1],
                                          file_date=datetime.now(),
                                          file_size=str(file.size / 1000) + "K",
                                          file_path='/Data/')
                    destination.close()
    files = TbFile.objects.all()
    return render(request, "index.html", {'files': files})


def creat(request, select_list):
    if request.method == "POST":
        num = request.POST['Num']
        if num == "" or select_list == "null":
            error_msg = "剧本字数或剧本选择不能为空"
            result = ""
        else:
            id_list = list(map(int, str(select_list).split(",")))
            data_list = TbFile.objects.filter(file_id__in=id_list)

            '''
            说明：在此处调用剧本续写的函数 
            参数 num（用户要生成的剧本字数） 数据类型为字符串 如果需要整型可以转换一下
            参数 data_list（用户选择的剧本数据源）数据类型为对象集合 使用for in 可以遍历data_list的对象 这样就可以获取到路径
            返回 可以是字符串或路径 如果是字符串可以将其赋值给result 如果是路径可以根据路径读文件后赋值给result
            '''
            error_msg = "剧本续写执行完毕"
            result = "剧本续写新内容！"
    files = TbFile.objects.all()
    return render(request, "index.html", {'files': files, 'result': result, 'error_msg': error_msg})


def save(request):
    if request.method == "POST":
        error_msg = "剧本保存执行完毕"
    files = TbFile.objects.all()
    return render(request, "index.html", {'files': files, 'error_msg': error_msg})


def clear(request):
    if request.method == "POST":
        error_msg = "剧本清空执行完毕"
    files = TbFile.objects.all()
    return render(request, "index.html", {'files': files, 'error_msg': error_msg})


def scriptmanage(request):
    files = TbFile.objects.all()
    return render(request, "scriptmanage.html", {'files': files})


def permissionmanage(request):
    files = TbFile.objects.all()
    return render(request, "permissionmanage.html", {'files': files})


def rolemanage(request):
    files = TbFile.objects.all()
    return render(request, "rolemanage.html", {'files': files})


def usermanage(request):
    users = TbUser.objects.all()
    return render(request, "usermanage.html", {'users': users})


def page_not_found(request):
    return render(request, "404.html")


def page_inter_error(request):
    return render(request, "500.html")
