import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--val_perc', action='store', dest='val_perc', help='Percentage for validation')
parser.add_argument('-s', '--src_dir', action='store', dest='src_dir', help='Dir with examples')


args = parser.parse_args()
val_perc = args.val_perc
src_dir = args.src_dir

if not val_perc or not src_dir:
    print(parser.print_help())
    exit(-1)

src_dir =  src_dir[:-1] if src_dir[-1] =='/' else src_dir
print(src_dir)
validate_dir = src_dir + '_validate'
if (not os.path.exists(validate_dir)):
    os.mkdir(validate_dir)
for sub_dir in os.listdir(src_dir):
    full_sub_dir = os.path.join(src_dir, sub_dir)
    if os.path.isdir(full_sub_dir):
        dir_files = os.listdir(full_sub_dir)
        files_num = len([name for name in dir_files
            if os.path.isfile(os.path.join(full_sub_dir, name))])
        val_num = files_num * float(val_perc)
        print('Will move {0} files out of {1} from {2} folder'.format(
            val_num, files_num, full_sub_dir
        ))
        full_validate_sub_dir = os.path.join(validate_dir, sub_dir)
        if not os.path.exists(full_validate_sub_dir):
            os.mkdir(full_validate_sub_dir)
        i = 0
        for curr_file in dir_files:
            file_full_path = os.path.join(full_sub_dir, curr_file)
            file_new_full_path = os.path.join(full_validate_sub_dir, curr_file)
            os.rename(file_full_path, file_new_full_path)
            i += 1
            if i >= val_num:
                break


        



