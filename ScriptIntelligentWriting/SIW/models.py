from django.db import models


# Create your models here.


class TbFile(models.Model):
    file_id = models.AutoField(primary_key=True)
    file_name = models.CharField(max_length=255)
    file_type = models.CharField(max_length=10, blank=True, null=True)
    file_date = models.DateTimeField(blank=True, null=True)
    file_size = models.CharField(max_length=50, blank=True, null=True)
    file_path = models.CharField(max_length=255)

    class Meta:
        managed = False
        db_table = 'tb_file'


class TbUser(models.Model):
    user_id = models.CharField(primary_key=True, max_length=50)
    user_name = models.CharField(max_length=50)
    user_password = models.CharField(max_length=50)
    user_description = models.CharField(max_length=255, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'tb_user'
