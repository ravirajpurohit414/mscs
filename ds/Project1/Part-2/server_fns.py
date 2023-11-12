import base64
import shutil
import logging
import os

logging.basicConfig(format="SERVER:%(asctime)s: %(message)s", 
                    level=logging.INFO, 
                    datefmt="%H:%M:%S")

root = os.path.dirname(os.path.abspath(__file__))
server_list_of_files = os.path.join(root, "server_files")
client_path = os.path.join(root, "client_files")

class Server:
    def get_server_files(self):
        """
        Retrieve a list of file paths relative to the 'server_files' directory by recursively exploring a directory.

        Returns
        -------
        list: A list of file paths relative to the 'server_files' directory.
        """
        # Initialize an empty list to store the file paths.
        list_of_files = []

        # Recursively traverse the directory structure starting from client_path.
        for root, _, files in os.walk(server_list_of_files, topdown=False):
            for filename in files:
                # Create the full file path.
                file_path = os.path.join(root, filename)

                # Extract the relative path by removing everything before 'server_files/'.
                base_path = file_path[file_path.rindex('server_files')+len("server_files/"):]

                # Add the relative path to the list.
                list_of_files.append(base_path)

        # Return the list of relative file paths.
        return list_of_files
        
    def download(self, file_path):
        """
        Download and retrieve the content of a file.

        Args
        ----
        file_path (str): The path to the file to be downloaded.

        Returns
        -------
        bytes or str: The content of the downloaded file as bytes or an error message as a string.
        """
        try:
            # Construct the full source download path.
            src_dwn_path = os.path.join(server_list_of_files, file_path)

            # Get the content of the file using the 'get_file_content' function.
            data = self.get_file_content(src_dwn_path)

            return data
        except Exception as e:
            # Log any errors that occur during the download process.
            logging.error("------- Error in downloading -------")
            return "------- Error in downloading -------"
        
    def rename(self, old_name, new_name):
        """
        Rename a file or directory from the old name to the new name.

        Args
        ----
        old_name (str): The current name of the file or directory.
        new_name (str): The desired new name for the file or directory.

        Returns
        -------
        bool: True if renaming is successful, False otherwise.
        """
        # Construct the full paths for the old and new names.
        cl_re_og = os.path.join(client_path, old_name)
        cl_re_new = os.path.join(client_path, new_name)
        src_re_og = os.path.join(server_list_of_files, old_name)
        src_re_new = os.path.join(server_list_of_files, new_name)

        try:
            # Log the renaming operation.
            logging.info(f"------ Renaming from {cl_re_og} to {cl_re_new} ------")

            # Rename both client and server-side files or directories.
            os.rename(src_re_og, src_re_new)
            os.rename(cl_re_og, cl_re_new)

            logging.info("------ Renaming successful -------")
            return True
        except OSError as e:
            # Log any errors that occur during the renaming process.
            logging.error(e)
            return False
        
    def upload(self, file_object, path):
        """
        Upload a file or directory to the specified path.

        Args
        ----
        file_object (dict): A dictionary representing the file or directory to be uploaded.
        path (str): The destination path where the file or directory should be uploaded.

        Returns
        -------
        bool: True if the upload is successful, False otherwise.
        """
        try:
            keyword = "client_files"

            if keyword in path:
                # Extract the target path relative to 'client_files/'.
                target_path = path.split("client_files/", 1)[1]
                target_path_abs = os.path.join(server_list_of_files, target_path)
            else:
                # Use the provided path as is.
                target_path_abs = os.path.join(server_list_of_files, path)

            if file_object['is_folder'] and not os.path.exists(target_path_abs):
                # Create the directory if it doesn't exist for folder uploads.
                os.makedirs(target_path_abs)
                return False
            elif file_object['is_folder']:
                return False

            # Upload the file to the target path.
            self.upload_file(file_object, target_path_abs)
            return True
        except Exception as e:
            # Log any errors that occur during the upload process.
            logging.error("------- Error in uploading -------")
            return False       
    
    def cleanup(self):
        """
        Cleanup the contents of the specified directory.

        Returns
        -------
        bool: True if cleanup is successful, False otherwise.
        """
        for f in os.listdir(server_list_of_files):
            file_path = os.path.join(server_list_of_files, f)
            try:
                if os.path.islink(file_path) or os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                # Log any errors that occur during the cleanup process.
                logging.error("------ Error in deletion ------")
                return False
        return True      
        
    def delete(self, file_path):
        """
        Delete a file or directory located at the specified path.

        Args
        ----
        file_path (str): The path to the file or directory to be deleted.

        Returns
        -------
        bool: True if deletion is successful, False otherwise.
        """
        try:
            keyword = "client_files"

            # Check if the 'client_files' keyword is in the file_path.
            if keyword in file_path:
                target_path = file_path.split("client_files/", 1)[1]
            else:
                target_path = file_path

            # Construct the absolute target path.
            target_path_abs = os.path.join(server_list_of_files, target_path)

            if os.path.isfile(target_path_abs):
                # Delete the file if it exists.
                os.remove(target_path_abs)
            else:
                # Delete the directory and its contents if it exists.
                shutil.rmtree(target_path_abs)

            return True
        except OSError as e:
            # Log any errors that occur during the deletion process.
            logging.error("------ Error while deletion ------")
            return False
    
    def upload_file(self, file_object, target_path_abs):
        """
        Upload a file to the specified target path.

        Args
        ----
        file_object (dict): A dictionary representing the file to be uploaded, including 'data'.
        target_path_abs (str): The absolute path where the file should be uploaded.

        Returns
        -------
        bool: True if the upload is successful, False otherwise.
        """
        try:
            # Extract the directory path from the target path.
            base_path = target_path_abs[:target_path_abs.rindex('/')]

            # Get the file content from the 'data' attribute in the file_object.
            content = file_object['data']
            
            # Ensure that the directory exists; create it if it doesn't.
            if not os.path.exists(base_path):
                os.makedirs(os.path.dirname(target_path_abs), exist_ok=True)
            
            # Write the base64-decoded content to the target path.
            with open(target_path_abs, 'wb') as f:
                f.write(base64.b64decode(content.data))

            return True
        except Exception as e:
            # Log any errors that occur during the upload process.
            print(e)
            logging.error("------ Error in uploading file ------")
            return False

    def get_file_content(self, file_path):
        """
        Read and encode the content of a file as base64.

        Args
        ----
        file_path (str): The path to the file whose content should be read and encoded.

        Returns
        -------
        bytes: The base64-encoded content of the file.
        """
        # Open the file, read its content, and then close the file.
        file = open(file_path, "rb")
        file_content = file.read()
        file.close()

        # Encode the file content in base64 and return it.
        return base64.b64encode(file_content)