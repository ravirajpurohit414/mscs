import os
import time
import base64
import filecmp
import threading
from xmlrpc.client import ServerProxy

from os.path import exists

server_proxy = ServerProxy('http://localhost:3000', verbose=False)
root = os.path.dirname(os.path.abspath(__file__))
s_path = os.path.join(root, "server_files/files")
c_path = os.path.join(root, "client_files/files")

def checkModified(path_to_server, path_to_client):
    source = filecmp.dircmp(path_to_server, path_to_client)
    for source in source.common_files:
        dest = "files/" +source
        src_abs_path, dest_full_path = os.path.join(c_path, source),  os.path.join(s_path, source)
        
        if exists(src_abs_path and dest_full_path):
            file_source_obj = gen_meta_data(root, src_abs_path, False)
            file_dest_obj = gen_meta_data(root, dest_full_path, False)
            file_source= get_content(src_abs_path,file_source_obj)
            file_dest= get_content(dest_full_path,file_dest_obj)
            if((round(get_timestamp(file_source)),1) != (round(get_timestamp(file_dest)),1)):
                is_uploaded = server_proxy.upload(file_source, dest)
                if is_uploaded:
                    print("------ modified file synced ------")
                else:
                    print("------ error in path : modify ------ ")


def gen_meta_data(root, name, is_folder):
    file_object = {
        'created_at': os.path.getctime(os.path.join(root, name)),
        'modified_at': os.path.getmtime(os.path.join(root, name)),
        'is_folder': is_folder,
        'file_path': os.path.join(root, name),
        'size': os.path.getsize(os.path.join(root, name))
    }
    return file_object


def get_content(path, file_object):
    if file_object['is_folder']:
        return file_object
    else:        
        txt_doc = open(path, "rb")
        data = txt_doc.read()
        txt_doc.close()
        file_object['data'] = base64.b64encode(data)
        return file_object

def get_timestamp(file_object):
    return file_object['modified_at']
    

def update_folder_uploads(path_to_server, path_to_client):    
    source = filecmp.dircmp(path_to_server, path_to_client)
    for source in source.right_only:
        dest = "files/"+source
        src_abs_path = os.path.join(c_path, source)
        if exists(src_abs_path):
            file_source_obj = gen_meta_data(root, src_abs_path, False)
            file_source= get_content(src_abs_path,file_source_obj)
            is_uploaded = server_proxy.upload(file_source, dest)
            if is_uploaded:
                print("------ new file synced ------")
            else:
                print("------ error in path : upload ------ ")


def update_folder_downloads(path_to_client, path_to_server):
    result = filecmp.dircmp(path_to_client,path_to_server)
    for x in result.left_only:
        dest = "files/" + x
        if result =='SYNC':
            continue
        else:
            server_proxy.delete(dest)
            print("------ file deleted ------")

def main():
    ## entry point function
    while True:
        thread2 = threading.Thread(target=update_folder_downloads, 
                                   args=(s_path, c_path))
        thread2.start()
        thread2.join()
        
        thread1 = threading.Thread(target=update_folder_uploads, 
                                   args=(s_path, c_path))
        thread1.start()
        thread1.join()
        
        thread3 = threading.Thread(target=checkModified, 
                                   args=(s_path, c_path))
        thread3.start()
        thread3.join()
        
        ## sleeping for 2 seconds
        for i in range(5):
            time.sleep(2)

if __name__ == '__main__':
   main()
   