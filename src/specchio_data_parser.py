"""
This file contains methods to attempt to parse horrible Specchio data into some coherent format.

This is only used in the case that reflectance and transmittance measurements have to be
loaded in separate files in separate folders one by one from specchio.ch web interface. The code is
a mess but one should not have to use this often.

"""

import csv
import os
import numpy as np
import toml


main_folder = os.path.normpath('../../SpeccioData')

def make_target( wls, r_m, t_m, path, sample_idx):

    if len(wls) != len(r_m) or len(wls) != len(t_m):
        raise ValueError(f'Length of the lists of wavelenghts ({len(wls)}), reflectances ({len(r_m)}) or transmittances ({len(t_m)}) did not match.')

    target_data = np.zeros((3, len(wls)))
    wls = np.array(wls)
    r_m = np.array(r_m).clip(0.,1.)
    t_m = np.array(t_m).clip(0.,1.)
    target_data[0] = wls
    target_data[1] = r_m
    target_data[2] = t_m
    target_data = np.transpose(target_data)
    floated_list = [[float(a), float(b), float(c)] for (a, b, c) in target_data]
    res = {'wlrt': floated_list}
    file_path = os.path.join(path, f'target_{sample_idx}.toml')
    with open(file_path, 'w+') as file:
        toml.dump(res, file)

def combine_pairs():
    pair_list = collect_pairs()
    for i,pair in enumerate(pair_list):
        sample_refl = pair[0]
        sample_tran = pair[1]
        if not sample_refl['is_reflectance']:
            temp = sample_refl
            sample_refl = sample_tran
            sample_tran = temp
        wls = sample_refl['wls']
        r_m = np.array([r for _, r in sorted(zip(wls, sample_refl['values']))])
        t_m = np.array([t for _, t in sorted(zip(wls, sample_tran['values']))])
        make_target(wls, r_m, t_m, main_folder, i)

def collect_pairs():
    sample_dict_list = open_files()
    pair_count = 0
    pair_list = []
    for i,sample_dict in enumerate(sample_dict_list):
        sample_id = sample_dict['sample_id']
        is_adaxial = sample_dict['is_adaxial']
        is_shaded = sample_dict['is_shaded']
        is_reflectance = sample_dict["is_reflectance"]
        for j in range(i + 1, len(sample_dict_list)):
            sample_dict_other = sample_dict_list[j]
            sample_id_other = sample_dict_other['sample_id']
            is_adaxial_other = sample_dict_other['is_adaxial']
            is_shaded_other = sample_dict_other['is_shaded']
            is_reflectance_other = sample_dict_other["is_reflectance"]
            if sample_id == sample_id_other and is_adaxial == is_adaxial_other and is_shaded == is_shaded_other and is_reflectance != is_reflectance_other:
                # print(f'I found a pair of samples:')
                # print(f'sample {sample_id} is reflectance == {sample_dict["is_reflectance"]}')
                # print(f'sample {sample_id_other} is reflectance == {sample_dict_other["is_reflectance"]}')
                pair_count += 1
                pair_list.append([sample_dict, sample_dict_other])

    print(f'All in all {pair_count} pairs were found')
    return pair_list


def open_files():
    print(f'Trying to open this shitstorm...')

    sample_dict_list = []
    for subfolder in os.listdir(main_folder):
        # print(f'subfolder: "{os.path.join(main_folder, subfolder)}"')
        for filename in os.listdir(os.path.join(main_folder, subfolder)):
            file_path = os.path.join(main_folder, subfolder, filename)
            # print(f'\tfilepath: "{file_path}"')
            with open(file_path, newline='') as csv_file:
                reader = csv.reader(csv_file)
                full_dict = {}
                metadata = {}
                wls = []
                values = []

                for line in reader:
                    # print(f'\t\t{line}')
                    if len(line)==0:
                        continue
                    key = line[0]
                    value = line[1]
                    # print(f'"{key}":{value}')
                    try:
                        # try casting key to float, which will succeed for wavelengths and fail for metadata
                        wls.append(float(key))
                        values.append(float(value))
                    except ValueError as e:
                        # print(f'"{key}":{value}')
                        metadata[key] = value

                filename = metadata['File Name']
                part = filename.rsplit('_')
                is_mean = 'mean' == part[-1]
                if not is_mean:
                    print(f'File {filename} is not mean file. Skipping...')
                    continue

                is_reflectance = 'reflectance' == part[len(part)-2]
                sample_id = part[1]
                is_shaded = 'S' == part[2]
                is_adaxial = 'A.xls' == part[3]

                full_dict['is_reflectance'] = is_reflectance
                full_dict['sample_id'] = sample_id
                full_dict['is_shaded'] = is_shaded
                full_dict['is_adaxial'] = is_adaxial
                full_dict['meta_data'] = metadata
                full_dict['wls'] = wls
                full_dict['values'] = values
                # print(full_dict)
                sample_dict_list.append(full_dict)

    return sample_dict_list
