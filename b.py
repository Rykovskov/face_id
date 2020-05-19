#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import datetime
import sys
from pathlib import Path
import paramiko
import smtplib
import logging
import traceback


#Define constant
#Time
storage_time = 6 #Month
#Servers
list_Servers = [
                {'NameServer':'Server1',
               'IP': "192.168.21.48",.
               'user': 'max',.
               'pass': 'tv60hu02',.
               'DirBackup':['/var/log/nginx',
                            '/var/log/app'],
               'WorkDir':'/home/max/test'},
                {'NameServer':'Server2',
               'IP': "192.168.21.47",.
               'user': 'max',.
               'pass': 'tv60hu02',
               'DirBackup':['/var/log/nginx',
                            '/var/log/app'],
               'WorkDir':'/home/max/test1'}
               ]
#Info
logging.basicConfig(filename="Backup.log", level=logging.INFO)

#Other
min_free_space = 100 # In Mbyte

#Check free space on buckup server
def check_space(Server):
    st = os.statvfs(Server['WorkDir'])
    free_space = int(st.f_bsize * st.f_bavail / 1024 / 1024)
    if free_space<min_free_space:
       logging.error("Check space on the backup store device !!!!")
       return 0
    logging.info("Free space = " + str(free_space))
    return 1


#Functions for download files from remote server
def get_files(list_files, dir_name, local_dir, srv): # list_files - List files candidate to download dir_name - Directory name on remote server srv - Remote server
   try:
      client = paramiko.SSHClient()
      client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
      client.connect(srv['IP'],  username = srv['user'], password = srv['pass'])
      ftp=client.open_sftp()
      ftp.chdir(dir_name)
      for f in list_files:
          if not check_space(server):
             logging.error("Error befor get file " + f + " from server " + srv['IP'] + ". No space on storage!!! ")
             sys.exit()
          patch_to_file = os.path.join(local_dir, f)
          ftp.get(f, patch_to_file)
          if not os.path.exists(patch_to_file):
             logging.error("Error get file " + f + " from server  "+ srv['IP'])
      return 1
   except:
      logging.error("Error get file from server  " + srv['IP'])
      return 0


# Functions for remove name of files from list files
def remove_exsist_file_name(list_files, name_dir): #listfiles - List files from remote server name_dir  - Directory where wil be locations this files.
    #Remove name files where exist in local storage
    for f in list_files:
        fullpath = os.path.join(name_dir , f)
        if os.path.exists(fullpath):
           list_files.remove(f)
    return list_files


#Start code
yesterday_dt = datetime.datetime.now() - datetime.timedelta(days=1)

for server in list_Servers:
    for logdir in server['DirBackup']:
        print(logdir)
        print("--------------------------")
        if not check_space(server):
           sys.exit()
        path_app_log = os.path.join(server['WorkDir'] , logdir.split('/')[-1]) #Directory where savieng backups
        files = get_list_files_from_server(yesterday_dt, logdir, server)         #Get list files from server
        print('F1')
        print(files)
        if len(files)>0:
           get_files_list_name = remove_exsist_file_name(files, path_app_log)
        else:
           continue
        print('F2')
        print(get_files_list_name)
        if len(get_files_list_name)==0:
           logging.info("No candidate files to backups on " + server['IP'])
        #Get files from server
        get_files(get_files_list_name, logdir, path_app_log, server) #Get files from server
        if not check_space(server):
           sys.exit()
    sys.exit()
