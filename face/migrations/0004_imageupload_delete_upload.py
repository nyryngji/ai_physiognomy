# Generated by Django 5.0.6 on 2024-06-04 03:25

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('face', '0003_upload_delete_post'),
    ]

    operations = [
        migrations.CreateModel(
            name='ImageUpload',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='media/')),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.DeleteModel(
            name='Upload',
        ),
    ]