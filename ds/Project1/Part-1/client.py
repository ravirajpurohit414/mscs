import os
import base64
import logging
import argparse

from xmlrpc.client import ServerProxy

## configure logging module 
format = "|-------- Client:%(asctime)s: %(message)s --------|"
logging.basicConfig(format=format, datefmt="%H:%M:%S", level=logging.INFO)

## setup required paths
root_path = os.path.dirname(os.path.abspath(__file__))
client_path = os.path.join(root_path, "client_files")

## setup a server proxy
server_proxy = ServerProxy('http://localhost:3000', verbose=False)

def get_client_files(client_path):
    """
    Retrieve a list of file paths relative to the 'client_files' directory by recursively exploring a directory.

    Args
    ----
    client_path (str): The root directory path where client files are located.

    Returns
    -------
    list: A list of file paths relative to the 'client_files' directory.
    """

    # Initialize an empty list to store the file paths.
    list_of_files = []

    # Recursively traverse the directory structure starting from client_path.
    for root, _, files in os.walk(client_path, topdown=False):
        for filename in files:
            # Create the full file path.
            file_path = os.path.join(root, filename)

            # Extract the relative path by removing everything before 'client_files/'.
            base_path = file_path[file_path.rindex('client_files')+len("client_files/"):]

            # Add the relative path to the list.
            list_of_files.append(base_path)

    # Return the list of relative file paths.
    return list_of_files


def get_file_data(file_path, file_object):
    """
    Retrieve and encode the contents of a file, if it's not a folder.

    Args
    ----
    path (str) : path of the file/folder.
    file_object (dict): A dictionary representing file metadata.

    Returns
    -------
    dict: The input 'file_object' dictionary updated with 'data' if it's not a folder.
    """
    if file_object['is_folder']:
        return file_object
    # Else open the file, read its contents, and then close the file.
    text_file = open(file_path, "rb")
    data = text_file.read()
    text_file.close()
    
    # Encode the file contents in base64 and add them to the file_object.
    file_object['data'] = base64.b64encode(data)
    return file_object

def gen_meta_data(root, name, is_folder):
    """
    Constructs a metadata dictionary for a given file or folder.

    Args
    ----
    root (str) : parent path of directory
    name (str) : name of the file/folder
    is_folder (bool): is the path is a directory

    Returns
    -------
    dict: A dictionary containing metadata information, including:
        - 'file_path' (str): The full path to the file or directory.
        - 'size' (int): The size of the file in bytes (0 for directories).
        - 'created_at' (float): The creation timestamp of the file or directory.
        - 'last_modified' (float): The last modification timestamp of the file or directory.
        - 'is_folder' (bool): Indicates if it's a directory or not.
    """

    root_name = os.path.join(root, name)
    file_object = {
        'file_path': root_name,
        'size': os.path.getsize(root_name),
        'created_at': os.path.getctime(root_name),
        'last_modified': os.path.getmtime(root_name),
        'is_folder': is_folder
    }

    return file_object

def download_file(data, file_path):
    """
    Create a file at the specified path with the given data.

    Args
    ----
    data (str): The data to write to the file, expected to be base64-encoded.
    file_path (str): The path where the file should be created.

    Returns
    -------
    bool: True if the file creation was successful, False otherwise.
    """
    try:
        # Create the full path by joining 'client_path' and the provided 'path'.
        path = os.path.join(client_path, file_path)

        # Ensure that the parent directory exists.
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

        # Decode the base64 data and write it to the file.
        with open(path, 'wb') as f:
            print("-----1234 \n", f)
            f.write(base64.b64decode(data.data))

        return True
    except Exception as ex:
        # Log any exceptions that occur during the file creation.
        logging.error(f"------ Exception occurred: {ex} ------")
        return False

if __name__ == '__main__':
    ## Entry point function

    ## initialize a argument parser
    argument_parser = argparse.ArgumentParser()

    ## assign the arguments
    argument_parser.add_argument("-ls", "--list", type=str, nargs=1, metavar=('folderpath'))
    argument_parser.add_argument("-up", "--upload", type=str, nargs=2, metavar=('source', 'destination'))
    argument_parser.add_argument("-dow", "--download", type=str, nargs=2, metavar=('source', 'destination'))
    argument_parser.add_argument("-ren", "--rename", type=str, nargs=2, metavar=('old-name', 'rename-to'))
    argument_parser.add_argument("-del", "--delete", type=str, nargs=1, metavar=('filepath'))
    
    args = argument_parser.parse_args()

    if args.list is not None:
        entity = args.list[0]
        if entity == "server":
            server_list_of_files = server_proxy.get_server_files()
            logging.info(server_list_of_files)
        elif entity == "client":
            client_list_of_files = get_client_files(client_path=client_path)
            logging.info(client_list_of_files)
        else:
            logging.error("----- Verify Path -----")

    elif args.upload is not None:
        src, dest = args.upload[0], args.upload[1]
        src_path = os.path.join(client_path, src)
        if os.path.exists(src_path):
            file_src_obj = gen_meta_data(root_path, src_path, False)
            
            if not file_src_obj['is_folder']:
                file_src = get_file_data(src_path, file_src_obj)
                is_uploaded = server_proxy.upload(file_src, dest)
                if is_uploaded:
                    print("------- Upload Successful ------")
                else:
                    print("------- Upload Unsuccessful -------")
            else:
                logging.error("------ File not found ", src_path, " ------")
                print("------- Error: Folder found instead of a file:", file_src_obj["file_path"], " -------")
        else:
            logging.error("------ File not found ", src_path, " ------")
        
    elif args.download is not None:
        src, dest = args.download[0], args.download[1]
        data = server_proxy.download(src)
        if(data == 'DownloadException'):
            logging.info("------- File Unavailable -------")
        else:
            if(download_file(data, dest)):
                logging.info("------- Download Successful -------")

    elif args.rename is not None:
        old_filename, new_filename = args.rename[0], args.rename[1]
        if server_proxy.rename(old_filename, new_filename):
            logging.info("------- File Renaming Successful -------")

    elif args.delete is not None:
        target_path = args.delete[0]
        if server_proxy.delete(target_path):        
            logging.info("------- File Deletion Successful -------")
        else:
            logging.error("------- File not found -------")

