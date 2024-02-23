# coding:--utf-8--
import numpy as np
import re

TOTAL_ITEMS = 624961


# 判别是否为 user_id|item_num。
def is_user_item(_str):
    reg_exp = re.compile(r"[0-9]*[|][0-9]*")
    return reg_exp.match(_str)


# 读取训练集。
def get_train_data():
    user_item_map = dict()
    user_id = 0
    all_items_mean = 0
    items_num = 0
    each_item_mean = dict()
    each_item_devia = dict()
    each_item_count = np.zeros(624961)

    file_train = open('./train.txt')
    line = file_train.readline()

    while line:
        if is_user_item(line):
            user_item = line.split('|')
            user_id = int(user_item[0])
            # print("user_id:", user_id, "numOfItem:", numOfItem)
        else:
            item_score = line.split('  ')
            item_id = int(item_score[0])
            score = float(item_score[1])
            # print("item_id:", item_id, "score:", score)

            each_item_mean.setdefault(item_id, 0)
            each_item_mean[item_id] += score
            each_item_count[item_id] += 1
            all_items_mean += score
            items_num += 1
            user_item_map = merge_two_dict(user_item_map, user_id, item_id, score)

        line = file_train.readline().strip()

    file_train.close()
    all_items_mean = all_items_mean / items_num

    # each_item_mean中的item均已被评分。
    for i in each_item_mean:
        ave_score = each_item_mean[i] / each_item_count[i]
        each_item_mean.update({i: ave_score})
        each_item_devia.update({i: ave_score - all_items_mean})

    return user_item_map, all_items_mean, each_item_mean, each_item_devia


# 添加到二维字典中
def merge_two_dict(res_dict, user, item, score):
    if user in res_dict:
        res_dict[user].update({item: score})
    else:
        res_dict.update({user: {item: score}})

    return res_dict


'''def save_data(user_item_map):
    for i in range(3):
        file_train = open("./user" + str(i) + ".txt", 'w')
        dict_user = user_item_map[i]
        
        for j in range(len(dict_user.keys())):
            file_train.write(str(dict_user.keys()[j]) + " " + str(dict_user.values()[j]) + '\n')
            
        file_train.close()
'''


# 分割成训练集、测试集。
def split_origin_dict(ori_dict, percentage):
    total_users = len(ori_dict.keys())
    trainset_num = int(total_users * (1 - percentage))
    user_item_map_train = dict()
    user_item_map_test = ori_dict.copy()

    for i in range(trainset_num):
        user_item_map_train.update({i: ori_dict[i]})
        user_item_map_test.pop(i)

    for u in user_item_map_test:
        ori_test = user_item_map_test[u]

        for item, score in list(ori_test.items()):
            if item < TOTAL_ITEMS / 2:
                user_item_map_train = merge_two_dict(user_item_map_train, u, item, score)
                user_item_map_test[u].pop(item)

    return user_item_map_train, user_item_map_test


# 读取测试数据
def get_test_data():
    file_test = open('./test.txt')
    line = file_test.readline().strip()

    res_test_dict = dict()
    user_id = 0

    while line:
        if is_user_item(line):
            temp = line.split('|')
            user_id = int(temp[0])
        else:
            item_id = int(line)
            res_test_dict = merge_two_dict(res_test_dict, user_id, item_id, 0)

        line = file_test.readline().strip()

    file_test.close()
    return res_test_dict


percentage = 0.2
user_item_map, all_items_mean, each_item_mean, each_item_devia = get_train_data()
'''f1 = open("user_item_map.txt", 'w')
f1.write(str(user_item_map))
f1.close()

f2 = open("all_items_mean.txt", 'w')
f2.write(str(all_items_mean))
f2.close()

f4 = open("each_item_devia.txt", 'w')
f4.write(str(each_item_devia))
f4.close()

keys = list(each_item_mean)
keys.sort()
print(keys[-1])
'''

user_item_map_train, user_item_map_test = split_origin_dict(user_item_map, percentage)

res_test_dict = get_test_data()
