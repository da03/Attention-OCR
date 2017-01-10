import sys

input_path = '/media/data2/sivankeret/Datasets/mnt/ramdisk/max/90kDICT32px/annotation_train.txt'
lex_path = '/media/data2/sivankeret/Datasets/mnt/ramdisk/max/90kDICT32px/lexicon.txt'
output_path = '/media/data2/sivankeret/Datasets/mnt/ramdisk/max/90kDICT32px/annotation_train_words.txt'

with open(lex_path,'r') as lex_f:
    all_words = lex_f.readlines()
word_dict = dict(enumerate(all_words))

with open(input_path,'r') as input_f:
    all_lines = input_f.readlines()

new_lines = [line.split(' ')[0] + ' ' + word_dict[int(line.split(' ')[1])] for line in all_lines]

with open(output_path, 'w') as out_f:
    out_f.writelines(new_lines)
