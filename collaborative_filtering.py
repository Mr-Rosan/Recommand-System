# coding:--utf-8--
import math
import time
from read_data import *


# similarity(user_item_map_train, user_test in user_item_map_test, other in user_item_map_train, perUserAverScore)
# Pearson correlation coefficient(用户u1与u2之间)
def computePCC(Dict, u1, u2, perUserAverScore):
    S_xy = {}

    for item in Dict.get(u1):
        if item in Dict.get(u2):
            S_xy[item] = 1

    lenOfSxy = len(S_xy)

    if lenOfSxy == 0:
        return 0

    u1_averScore = perUserAverScore[u1]
    u2_averScore = perUserAverScore[u2]
    r_xs = np.array([Dict[u1][item] for item in S_xy])
    r_ys = np.array([Dict[u2][item] for item in S_xy])
    mole = np.vdot((r_xs - u1_averScore), (r_ys - u2_averScore))
    den = computeEucDis(r_xs, u1_averScore) * computeEucDis(r_ys, u2_averScore)

    if den == 0:
        return 0
    sim_xy = mole / den
    return sim_xy


# 对每个用户的所有电影评分求均值和偏差。
def getPerUserAverSco(Dict, all_items_mean):
    averScore = dict()
    deviationOfUser = dict()

    for user in Dict:
        numOfItem = len(Dict[user].keys())
        sumOfScore = 0

        for score in Dict[user].values():
            sumOfScore += score

        averscore = float(sumOfScore) / numOfItem
        # print("################################",numOfItem)
        averScore.update({user: averscore})
        deviationOfUser.update({user: averscore - all_items_mean})

    return averScore, deviationOfUser


def computeEucDis(vec1, vec2):
    return math.sqrt(np.sum((vec1 - vec2) ** 2))


# topMatches(user_item_map_train, user_test)
# 计算用户之间的相似度
def topMatches(Dict, User, similarity=computePCC):
    sim_User = dict()
    # sim_User = [(similarity(Dict, User, other, perUserAverScore), other) for other in Dict if other != User]
    # 相似度小于零的相似用户就不再存储在sim_User中，从而较少计算复杂度与空间消耗
    for other in Dict:
        if other == User:
            continue
        sim = similarity(Dict, User, other, perUserAverScore)
        if sim < 0:
            continue
        else:
            sim_User.update({sim: other})
    sim_User = sorted(sim_User.items(), key=lambda item: item[0], reverse=True)
    return sim_User


def getRMSE(user_item_map_test, user_item_map_train, perUserAverScore, each_item_devia, deviationOfUser, all_items_mean):
    # predictDict_test=dict()
    RMSE = 0
    numOfR = 0

    for user_test in user_item_map_test:
        totals = {}
        simSums = {}
        similDict_user = topMatches(user_item_map_train, user_test)
        devia_user = deviationOfUser[user_test]

        for item in user_item_map_test[user_test]:
            totals.setdefault(item, 0)
            simSums.setdefault(item, 0)
            devia_item = each_item_devia[item]
            temp2 = all_items_mean + devia_item

            for simItem in similDict_user:
                simUser = simItem[1]

                if item not in user_item_map_train[simUser].keys():
                    continue
                devia_simUser = deviationOfUser[simUser]
                if simItem[0] <= 0:
                    break  # 如果与用户user_test的相似度是小于零的，则忽略不计算

                totals[item] += simItem[0] * (user_item_map_train[simUser][item] - (temp2 + devia_simUser))
                simSums[item] += simItem[0]

            # simSum[item]有可能为零
            if simSums[item] == 0:
                scorePredict = perUserAverScore[user_test]
            else:
                scorePredict = totals[item] / simSums[item] + temp2 + devia_user

            if scorePredict < 0:
                scorePredict = 0
            elif scorePredict > 100:
                scorePredict = 100

            RMSE += math.pow(user_item_map_test[user_test][item] - scorePredict, 2)
            print("scorePredict:", user_test, item, user_item_map_test[user_test][item], scorePredict)
            numOfR += 1

    RMSE = math.sqrt(RMSE) / numOfR
    return RMSE


def writePredictScoreToFile(user_item_map_test, user_item_map_train, perUserAverScore, each_item_devia, deviationOfUser):
    fr = open("./result.txt", "w")
    for user_test in user_item_map_test:
        fr.write(str(user_test) + '|6\n')
        totals = {}
        simSums = {}
        similDict_user = topMatches(user_item_map_train, user_test)
        devia_user = deviationOfUser[user_test]
        for item in user_item_map_test[user_test]:
            totals.setdefault(item, 0)
            simSums.setdefault(item, 0)
            if item not in each_item_devia.keys():
                scorePredict = perUserAverScore[user_test]
                fr.write(str(item) + '  ' + str(scorePredict) + '\n')
                continue
            else:
                devia_item = each_item_devia[item]
                # 如果该项没有被其他用户打过分，则在each_item_devia中该项不存在，那么就不能使用
                # 相似用户的评分对其进行计算，那么就使用该用户的平均打分作为该项的分数
                temp2 = all_items_mean + devia_item
                for simItem in similDict_user:
                    simUser = simItem[1]
                    devia_simUser = deviationOfUser[simUser]
                    if item not in user_item_map_train[simUser].keys():
                        continue
                    # 如果与用户user_test的相似度是小于零的，则忽略不计算
                    if simItem[0] <= 0:
                        break
                    totals[item] += simItem[0] * (user_item_map_train[simUser][item] - (temp2 + devia_simUser))
                    simSums[item] += simItem[0]
                # simSum[item]有可能为零
                if simSums[item] == 0:  # 如果该用户与所有其他用户的相似度都为0，则取该用户对所有项的平均值作为该项的评分
                    scorePredict = perUserAverScore[user_test]
                else:
                    scorePredict = totals[item] / simSums[item] + temp2 + devia_user
                # 不关心对于评分不高项的预测，那么对于真实分数低的项的评估误差就可以忽略不计，不计算到RMSE中

                if scorePredict < 0:
                    scorePredict = 0
                elif scorePredict > 100:
                    scorePredict = 100
                # print("scorePredict: ", scorePredict)
                fr.write(str(item) + '  ' + str(scorePredict) + '\n')
    fr.close()


# sim=computePCC(user_item_map_train,15867,0)
# similarityDict_userId=topMatches(user_item_map_train,0)

perUserAverScore, deviationOfUser = getPerUserAverSco(user_item_map_train, all_items_mean)
'''f1 = open("perUserAverScore.txt", 'w')
f1.write(str(perUserAverScore))
f1.close()'''

# 在训练数据集上测试RMSE
start = time.time()
RMSE = getRMSE(user_item_map_test, user_item_map_train, perUserAverScore, each_item_devia, deviationOfUser, all_items_mean)
end = time.time()

print('训练时长： %s 分钟' % ((end - start) / 60))
print('RMSE: ', RMSE)

start = time.time()
writePredictScoreToFile(res_test_dict, user_item_map, perUserAverScore, each_item_devia, deviationOfUser)
end = time.time()

print('写入时长 %s 分钟' % ((end - start) / 60))
