"""
Check directory structure and create missing directories.

"""

from src.data import path_handling as PH
from src.data import toml_handling as TH
from src import constants as C

def initialize():
    print("Hello, this is the initialization script.")

    check_directory_structure()



def check_directory_structure():

    # Read toml

    def read_entry(sub_dict: dict, dir_list):

        path_builder = dir_list
        entry_name = ""
        entry_type = ""
        sub_entry_type = ""
        expected_file_names = None
        new_sub_dict = None
        for key, value in sub_dict.items():
            # print(f"Key in read: {key}, Value: {value}")

            if key == "name":
                entry_name = value

            if key == "type":
                entry_type = value
            elif key == "expected_file_names":
                # print(f"Expected file names: {value}")
                expected_file_names = value

            if isinstance(value, dict) and value["type"] == "dir":
                sub_entry_type = value["type"]
                sub_entry_name = value["name"]
                new_sub_dict = value
                print("DEEPER")
                path_builder.append(sub_entry_name)
                read_entry(sub_dict=value, dir_list=path_builder)
                del path_builder[-1]
                # more_path = read_entry(sub_dict=value, dir_list=path_builder)

        if entry_type == "dir":

            current_path = PH.path_directory_project_root()
            if entry_name != "Root":
                # path_builder.append(entry_name)
                print(f"Path builder: {path_builder}")
                for dir_name in path_builder:
                    current_path = PH.join(current_path, dir_name)

            print(f"Current path: {current_path}")

            if expected_file_names is not None:
                print(f"Directory {entry_name} should contain files")
                for file_name in expected_file_names:
                    print(f"Expecting: {current_path}/{file_name}'")





    def_dir = PH.path_directory_definitions()
    struct_toml = TH.read_toml_as_dict(directory=def_dir, filename=C.file_directory_structure)
    print(struct_toml)

    for key, value in struct_toml.items():
        # print(f"Key: {key}, Value: {value}")
        if key == "root":
            # for dir_key, dir_value in value.items():
            #     print(f" Key: {dir_key}, Value: {dir_value}. Processing")
            read_entry(value, dir_list=[])
                # Check if directory exists
                # if not PH.check_directory_exists(dir_value):
                #     # Create directory if it doesn't exist
                #     PH.create_directory(dir_value)
                #     print(f"Created directory: {dir_value}")
                # else:
                #     print(f"Directory already exists: {dir_value}")

    # Check if directories exist
    # If not, create them