from django.shortcuts import render
from SIW.models import TbFile, TbUser
from datetime import datetime
import training


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
                                          file_type="" if len(str(file.name).split(".")) == 1 else
                                          str(file.name).split(".")[1],
                                          file_date=datetime.now(),
                                          file_size=str(file.size / 1000) + "K",
                                          file_path='/Data/')
                    destination.close()
    files = TbFile.objects.all()
    return render(request, "index.html", {'files': files})


def creat(request, select_list):
    if request.method == "POST":
        path = ""
        length = request.POST['Length']
        protagonist = request.POST['Protagonist']
        if length == "" or protagonist == "" or select_list == "null":
            error_msg = "剧本字数、剧本主角、剧本选择不能为空"
            result = ""
        else:
            id_list = list(map(int, str(select_list).split(",")))
            data_list = TbFile.objects.filter(file_id__in=id_list)
            for data in data_list:
                path += (data.file_path + data.file_name + "." + data.file_type + "|") if data.file_type != "" else (
                        data.file_path + data.file_name + data.file_type + "|")
            path = path[:-1]
            error_msg = "剧本续写执行完毕"
            result = training.generate(path, length, protagonist)
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
