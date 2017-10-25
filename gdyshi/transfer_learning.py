#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os.path
import random
import sqlite3
import numpy as np
import shutil
import tensorflow as tf
from tensorflow.python.platform import gfile
from PIL import Image

# Inception-v3模型瓶颈层的节点个数
BOTTLENECK_TENSOR_SIZE = 2048

# Inception-v3模型中代表瓶颈层结果的张量名称。
# 在谷歌提出的Inception-v3模型中，这个张量名称就是'pool_3/_reshape:0'。
# 在训练模型时，可以通过tensor.name来获取张量的名称。
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

# 图像输入张量所对应的名称。
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

COMMON_DIR = 'E:\data'
# COMMON_DIR = 'F:\\tmp\\bdci\\fangyi'

# 下载的谷歌训练好的Inception-v3模型文件目录
MODEL_DIR = COMMON_DIR + '\model'

# 下载的谷歌训练好的Inception-v3模型文件名
MODEL_FILE = 'tensorflow_inception_graph.pb'

# 因为一个训练数据会被使用多次，所以可以将原始图像通过Inception-v3模型计算得到的特征向量保存在文件中，免去重复的计算。
# 下面的变量定义了这些文件的存放地址。
CACHE_DIR = COMMON_DIR + '\\train\\bottleneck'
# 训练参数保存路径
PARM_FILE = COMMON_DIR + '\\train\parm\\variable.ckpt'
# 图片数据文件夹。
# 在这个文件夹中每一个子文件夹代表一个需要区分的类别，每个子文件夹中存放了对应类别的图片。
INPUT_DATA = COMMON_DIR + '\\train\\training'
INPUT_LABLE = COMMON_DIR + '\\train\lable'
TEMP_DATA_ALL = COMMON_DIR + '\\train\\tmp\\all'
TEMP_DATA_TRAIN = COMMON_DIR + '\\train\\tmp\\train'
TEMP_DATA_TEST = COMMON_DIR + '\\train\\tmp\\test'

# 验证的数据百分比
VALIDATION_PERCENTAGE = 10
# 测试的数据百分比
TEST_PERCENTAGE = 10

# 定义神经网络的设置
LEARNING_RATE = 0.005
STEPS = 40000
BATCH = 100

# 预处理数据集到数据库
def format_data_set():
    file_list = []
    c, conn = create_db('test.db')

    # 获取当前目录下所有的标记文件
    g = os.walk(INPUT_LABLE)
    # print(g)
    for paths, _, files in g:
        for file in files:
            filename = os.path.join(paths, file)
            file_list.append(filename)

    for file in file_list:
        f = open(file)  # 返回一个文件对象
        line_num = 0
        line = f.readline()  # 调用文件的 readline()方法
        # 通过文件内容获取类别的名称。
        while line:
            line_num = line_num + 1
            listFromLine = line.lstrip().rstrip().split(' ')
            # 检查label是否是6段
            if (6 != len(listFromLine)):
                print('fatal error: label not 6:' + file + 'but ' + str(len(listFromLine)) + 'line:' + str(line_num))
                print(listFromLine[6])
                exit(-1)
            # 检查文件名与label是否一致
            if -1 == file.find(listFromLine[1]):
                print('fatal error: label not matched file name:' + file + 'line:' + str(line_num))
                exit(-1)
            classify = listFromLine[1]
            # 检查对应图片是否存在
            image_file = get_real_img_name(listFromLine[0])
            if False == os.path.exists(image_file):
                print('fatal error: image:' + image_file + ' not found from label:' + file + 'line:' + str(line_num))
                exit(-1)
            image_id = update_image_table(c, image_file)
            classify_id = update_classify_table(c, classify)
            # # 拷贝有效图片
            # to_image_file = image_file.replace(INPUT_DATA, TEMP_DATA_ALL)
            # if not os.path.exists(os.path.dirname(to_image_file)):
            #     os.makedirs(os.path.dirname(to_image_file))
            # if not os.path.exists(to_image_file):
            #     print('copy file from ' + image_file + 'to' + to_image_file)
            #     shutil.copyfile(image_file, to_image_file)
            # to_label_file = file.replace(INPUT_LABLE, TEMP_DATA_ALL)
            # if not os.path.exists(to_label_file):
            #     print('copy file from ' + file + 'to' + to_label_file)
            #     shutil.copyfile(file, to_label_file)

            # 检查label标签是否超出图片大小
            img = Image.open(image_file)
            x, y = img.size
            lx = int(listFromLine[2])
            ly = int(listFromLine[3])
            rx = int(listFromLine[4])
            ry = int(listFromLine[5])
            if (x < int(listFromLine[2]) or x < int(listFromLine[4]) or y < int(listFromLine[3]) or y < int(
                    listFromLine[5])):
                print('error: label outofrange, img:' + file + 'line:' + str(line_num))
            if (x < lx):
                lx = x
            if (x < rx):
                rx = x
            if (y < ly):
                ly = x
            if (y < ry):
                ry = x
            # 检查label标签是否左上右下
            if (int(listFromLine[4]) < int(listFromLine[2]) or int(listFromLine[5]) < int(listFromLine[3])):
                print('error: label rect sec, img:' + file + 'line:' + str(line_num))
            if (lx > rx):
                temp = rx
                rx = lx
                lx = temp
            if (ly > ry):
                temp = ry
                ry = ly
                ly = temp
            insert_lable_table(c, classify_id, image_id, lx, ly, rx, ry)
            line = f.readline()
        f.close()
    create_dtset_from_db(c)
    conn.commit()
    conn.close()

# 创建训练集、验证集和数据集
def create_dtset_from_db(c):
    cursor = c.execute("SELECT id  from IMAGE_ALL")
    result = cursor.fetchall()
    for row in result:
        chance = np.random.randint(100)
        id = row[0]
        if chance < VALIDATION_PERCENTAGE:
            sqlstr = str("INSERT INTO IMAGE_VALID (IMAGE_ID)  VALUES (\"" + str(id) + "\" )")
        elif chance < (TEST_PERCENTAGE + VALIDATION_PERCENTAGE):
            sqlstr = str("INSERT INTO IMAGE_TEST (IMAGE_ID)  VALUES (\"" + str(id) + "\" )")
        else:
            sqlstr = str("INSERT INTO IMAGE_TRAIN (IMAGE_ID)  VALUES (\"" + str(id) + "\" )")
        c.execute(sqlstr)

# 插入数据到标签表中
def insert_lable_table(c, classify_id, image_id, lx, ly, rx, ry):
    sqlstr = str("INSERT INTO LABEL (IMAGE_ID,CLASSIFY_ID,LX,LY,RX,RY) \
                  VALUES (\"" + str(image_id) + "\", \"" + str(classify_id) + "\", " + str(lx) + ", " + str(
        ly) + ", " + str(rx) + ", " + str(ry) + " )")
    c.execute(sqlstr)

# 更新分类表，返回id
def update_classify_table(c, classify):
    select_sqlstr = str("SELECT id  from CLASSIFY WHERE NAME=\"" + classify + "\"")
    cursor = c.execute(select_sqlstr)
    result = cursor.fetchone()
    if result == None:
        insert_sqlstr = str("INSERT INTO CLASSIFY (NAME)  VALUES (\"" + classify + "\" )")
        cursor = c.execute(insert_sqlstr)
        cursor = c.execute(select_sqlstr)
        classify_id = cursor.fetchone()[0]
    else:
        classify_id = result[0]
    return classify_id

# 更新图片表，返回id
def update_image_table(c, image_file):
    select_sqlstr = str("SELECT id  from IMAGE_ALL WHERE PATH=\"" + image_file + "\"")
    cursor = c.execute(select_sqlstr)
    result = cursor.fetchone()
    if result == None:
        insert_sqlstr = str("INSERT INTO IMAGE_ALL (PATH)  VALUES (\"" + image_file + "\" )")
        cursor = c.execute(insert_sqlstr)
        cursor = c.execute(select_sqlstr)
        image_id = cursor.fetchone()[0]
    else:
        image_id = result[0]
    return image_id

# 创建数据库
def create_db(db_name):
    if os.path.exists(db_name):
        os.remove(db_name)
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('''CREATE TABLE IMAGE_ALL
           (ID INTEGER PRIMARY KEY     AUTOINCREMENT,
           PATH           TEXT    NOT NULL);''')
    c.execute('''CREATE TABLE CLASSIFY
           (ID INTEGER PRIMARY KEY     AUTOINCREMENT,
           NAME           TEXT    NOT NULL);''')
    c.execute('''CREATE TABLE LABEL
           (ID INTEGER PRIMARY KEY     AUTOINCREMENT,
           IMAGE_ID           INTEGER    NOT NULL,
           CLASSIFY_ID            INTEGER     NOT NULL,
           LX        INT NOT NULL,
           LY        INT NOT NULL,
           RX        INT NOT NULL,
           RY        INT NOT NULL);''')

    c.execute('''CREATE VIEW [VIEW_ALL]
            AS
            SELECT [IMAGE_ALL].[PATH] AS [IMAGE_ALL],
                   [CLASSIFY].[NAME] AS [CLASSIFY_NAME],
                   [LABEL].[LX],
                   [LABEL].[LY],
                   [LABEL].[RX],
                   [LABEL].[RY]
            FROM   [LABEL],
                   [IMAGE_ALL],
                   [CLASSIFY]
            WHERE  [LABEL].[IMAGE_ID] = [IMAGE_ALL].[ID]
                   AND [LABEL].[CLASSIFY_ID] = [CLASSIFY].[ID];''')
    c.execute('''CREATE VIEW [SET_ALL]
            AS
            SELECT [IMAGE_ALL].[ID] AS [IMAGE_ID],
                    [IMAGE_ALL].[PATH] AS [IMAGE_ALL],
                   [CLASSIFY].[ID] AS [CLASSIFY_ID],
                   [LABEL].[LX],
                   [LABEL].[LY],
                   [LABEL].[RX],
                   [LABEL].[RY]
            FROM   [LABEL],
                   [IMAGE_ALL],
                   [CLASSIFY]
            WHERE  [LABEL].[IMAGE_ID] = [IMAGE_ALL].[ID]
                   AND [LABEL].[CLASSIFY_ID] = [CLASSIFY].[ID];''')
    c.execute('''CREATE TABLE IMAGE_TRAIN
           (ID INTEGER PRIMARY KEY     AUTOINCREMENT,
           IMAGE_ID           INTEGER    NOT NULL);''')
    c.execute('''CREATE VIEW [VIEW_TRAIN]
            AS
            SELECT [IMAGE_ALL].[PATH] AS [IMAGE_ALL],
                   [CLASSIFY].[NAME] AS [CLASSIFY_NAME],
                   [LABEL].[LX],
                   [LABEL].[LY],
                   [LABEL].[RX],
                   [LABEL].[RY]
            FROM   [LABEL],
                   [IMAGE_ALL],
                   [IMAGE_TRAIN],
                   [CLASSIFY]
            WHERE  [LABEL].[IMAGE_ID] = [IMAGE_ALL].[ID]
                  AND [IMAGE_TRAIN].[IMAGE_ID] = [IMAGE_ALL].[ID]
                  AND [LABEL].[CLASSIFY_ID] = [CLASSIFY].[ID];''')
    c.execute('''CREATE VIEW [SET_TRAIN]
            AS
            SELECT [IMAGE_ALL].[ID] AS [IMAGE_ID],
                    [IMAGE_ALL].[PATH] AS [IMAGE_ALL],
                   [CLASSIFY].[ID] AS [CLASSIFY_ID],
                   [LABEL].[LX],
                   [LABEL].[LY],
                   [LABEL].[RX],
                   [LABEL].[RY]
            FROM   [LABEL],
                   [IMAGE_ALL],
                   [IMAGE_TRAIN],
                   [CLASSIFY]
            WHERE  [LABEL].[IMAGE_ID] = [IMAGE_ALL].[ID]
                  AND [IMAGE_TRAIN].[IMAGE_ID] = [IMAGE_ALL].[ID]
                  AND [LABEL].[CLASSIFY_ID] = [CLASSIFY].[ID];''')
    c.execute('''CREATE TABLE IMAGE_TEST
           (ID INTEGER PRIMARY KEY     AUTOINCREMENT,
           IMAGE_ID           INTEGER    NOT NULL);''')
    c.execute('''CREATE VIEW [VIEW_TEST]
            AS
            SELECT [IMAGE_ALL].[PATH] AS [IMAGE_ALL],
                   [CLASSIFY].[NAME] AS [CLASSIFY_NAME],
                   [LABEL].[LX],
                   [LABEL].[LY],
                   [LABEL].[RX],
                   [LABEL].[RY]
            FROM   [LABEL],
                   [IMAGE_ALL],
                   [IMAGE_TEST],
                   [CLASSIFY]
            WHERE  [LABEL].[IMAGE_ID] = [IMAGE_ALL].[ID]
                  AND [IMAGE_TEST].[IMAGE_ID] = [IMAGE_ALL].[ID]
                  AND [LABEL].[CLASSIFY_ID] = [CLASSIFY].[ID];''')
    c.execute('''CREATE VIEW [SET_TEST]
            AS
            SELECT [IMAGE_ALL].[ID] AS [IMAGE_ID],
                    [IMAGE_ALL].[PATH] AS [IMAGE_ALL],
                   [CLASSIFY].[ID] AS [CLASSIFY_ID],
                   [LABEL].[LX],
                   [LABEL].[LY],
                   [LABEL].[RX],
                   [LABEL].[RY]
            FROM   [LABEL],
                   [IMAGE_ALL],
                   [IMAGE_TEST],
                   [CLASSIFY]
            WHERE  [LABEL].[IMAGE_ID] = [IMAGE_ALL].[ID]
                  AND [IMAGE_TEST].[IMAGE_ID] = [IMAGE_ALL].[ID]
                  AND [LABEL].[CLASSIFY_ID] = [CLASSIFY].[ID];''')
    c.execute('''CREATE TABLE IMAGE_VALID
           (ID INTEGER PRIMARY KEY     AUTOINCREMENT,
           IMAGE_ID           INTEGER    NOT NULL);''')
    c.execute('''CREATE VIEW [VIEW_VALID]
            AS
            SELECT [IMAGE_ALL].[PATH] AS [IMAGE_ALL],
                   [CLASSIFY].[NAME] AS [CLASSIFY_NAME],
                   [LABEL].[LX],
                   [LABEL].[LY],
                   [LABEL].[RX],
                   [LABEL].[RY]
            FROM   [LABEL],
                   [IMAGE_ALL],
                   [IMAGE_VALID],
                   [CLASSIFY]
            WHERE  [LABEL].[IMAGE_ID] = [IMAGE_ALL].[ID]
                  AND [IMAGE_VALID].[IMAGE_ID] = [IMAGE_ALL].[ID]
                  AND [LABEL].[CLASSIFY_ID] = [CLASSIFY].[ID];''')
    c.execute('''CREATE VIEW [SET_VALID]
            AS
            SELECT [IMAGE_ALL].[ID] AS [IMAGE_ID],
                    [IMAGE_ALL].[PATH] AS [IMAGE_ALL],
                   [CLASSIFY].[ID] AS [CLASSIFY_ID],
                   [LABEL].[LX],
                   [LABEL].[LY],
                   [LABEL].[RX],
                   [LABEL].[RY]
            FROM   [LABEL],
                   [IMAGE_ALL],
                   [IMAGE_VALID],
                   [CLASSIFY]
            WHERE  [LABEL].[IMAGE_ID] = [IMAGE_ALL].[ID]
                  AND [IMAGE_VALID].[IMAGE_ID] = [IMAGE_ALL].[ID]
                  AND [LABEL].[CLASSIFY_ID] = [CLASSIFY].[ID];''')
    return c, conn


# 从数据库中读取所有的图片集合。
def create_image_sets():
    all_images = {}
    conn = sqlite3.connect('test.db')
    c = conn.cursor()

    validation_images = create_set(c, 'VALID')
    testing_images = create_set(c, 'TEST')
    training_images = create_set(c, 'TRAIN')
    conn.close()
    result = {
        'training': training_images,
        'validation': validation_images,
        'testing': testing_images,
        'all': all_images,
    }
    print('\ttotal_num:' + str(len(result['all'])))
    print('\ttraining_num:' + str(len(result['training'])))
    print('\tvalidation_num:' + str(len(result['validation'])))
    print('\ttesting_num:' + str(len(result['testing'])))
    return result


def create_set(c, set_name):
    images = {}
    sqlstr = str("SELECT IMAGE_ID  from IMAGE_" + set_name)
    cursor = c.execute(sqlstr)
    result = cursor.fetchall()
    for row in result:
        image_id = row[0]
        sqlstr = str(
            "SELECT IMAGE_ALL, CLASSIFY_ID, LX, LY, RX, RY from SET_" + set_name+" WHERE IMAGE_ID = \"" + str(image_id) + "\"")
        cursor1 = c.execute(sqlstr)
        result1 = cursor1.fetchall()
        for row1 in result1:
            path = row1[0]
            classfiy_id = row1[1]
            lx = row1[2]
            ly = row1[3]
            rx = row1[4]
            ry = row1[5]
            if path not in images:
                images[path] = [0, 0, 0, 0]
            if 1 == classfiy_id:
                images[path][0] = 1
            elif 2 == classfiy_id:
                images[path][1] = 1
            elif 3 == classfiy_id:
                images[path][2] = 1
            elif 4 == classfiy_id:
                images[path][3] = 1
                # images[path].append({'classfiy_id':classfiy_id,'lx':lx,'ly':ly,'rx':rx,'ry':ry})
    return images


# 替换标记文件中记录的图片路径为真实的图片路径
def get_real_img_name(old_path):
    new_path = ''
    if 0 == old_path.find('D:\CCF样本和标记工具\CCFModule'):
        new_path = old_path.replace('D:\CCF样本和标记工具\CCFModule', INPUT_DATA)
    elif 0 == old_path.find('E:\CCF样本'):
        new_path = old_path.replace('E:\CCF样本', INPUT_DATA)
    elif 0 == old_path.find('E:\CCFModule'):
        new_path = old_path.replace('E:\CCFModule', INPUT_DATA)
    elif 0 == old_path.find('E:\CCFModel'):
        new_path = old_path.replace('E:\CCFModel', INPUT_DATA)
    else:
        print('error replace path:' + old_path)
    return new_path


# 这个函数通过类别名称、所属数据集和图片编号获取一张图片的地址。
# image_lists参数给出了所有图片信息。
# image_dir参数给出了根目录。存放图片数据的根目录和存放图片特征向量的根目录地址不同。
# label_name参数给定了类别的名称。
# index参数给定了需要获取的图片的编号。
# category参数指定了需要获取的图片是在训练数据集、测试数据集还是验证数据集。
def get_image(image_lists, index):
    # 根据所属数据集的名称获取集合中的全部图片信息。
    mod_index = index % len(image_lists)
    # 获取图片的文件名。list(image_lists.keys())[label_index]
    image_full_name = list(image_lists.keys())[mod_index]
    ground_truth = image_lists[image_full_name]
    return image_full_name, ground_truth


# 这个函数通过类别名称、所属数据集和图片编号获取经过Inception-v3模型处理之后的特征向量文件地址。
def get_bottlenect(image_lists, index):
    # 获取图片的文件名。
    image_full_name, ground_truth = get_image(image_lists, index)
    bottlenect_full_name = image_full_name.replace(INPUT_DATA, CACHE_DIR)
    return bottlenect_full_name + '.txt', ground_truth


# 这个函数使用加载的训练好的Inception-v3模型处理一张图片，得到这个图片的特征向量。
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    # 这个过程实际上就是将当前图片作为输入计算瓶颈张量的值。这个瓶颈张量的值就是这张图片的新的特征向量。
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    # 经过卷积神经网络处理的结果是一个四维数组，需要将这个结果压缩成一个特征向量（一维数组）
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


# 这个函数获取一张图片经过Inception-v3模型处理之后的特征向量。
# 这个函数会先试图寻找已经计算且保存下来的特征向量，如果找不到则先计算这个特征向量，然后保存到文件。
def get_or_create_bottleneck(sess, image_lists, index, jpeg_data_tensor, bottleneck_tensor):
    # 获取一张图片对应的特征向量文件的路径。
    # label_lists = image_lists[image_name]
    bottleneck_full_name, ground_truth = get_bottlenect(image_lists, index)
    # print('bottleneck: ' + str(bottleneck_full_name))
    # print('ground_truth: ' + str(ground_truth))

    bottleneck_path = os.path.dirname(bottleneck_full_name)
    if not os.path.exists(bottleneck_path):
        os.makedirs(bottleneck_path)
    # 如果这个特征向量文件不存在，则通过Inception-v3模型来计算特征向量，并将计算的结果存入文件。
    if not os.path.exists(bottleneck_full_name):
        # 获取原始的图片路径
        image_full_name, _ = get_image(image_lists, index)
        # 获取图片内容。
        image_data = gfile.FastGFile(image_full_name, 'rb').read()
        # print(len(image_data))
        # 由于输入的图片大小不一致，此处得到的image_data大小也不一致（已验证），但却都能通过加载的inception-v3模型生成一个2048的特征向量。具体原理不详。
        # 通过Inception-v3模型计算特征向量
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        # 将计算得到的特征向量存入文件
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_full_name, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        # 直接从文件中获取图片相应的特征向量。
        with open(bottleneck_full_name, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    # 返回得到的特征向量
    return bottleneck_values, ground_truth


# 这个函数随机获取一个batch的图片作为训练数据。
def get_random_cached_bottlenecks(sess, n_classes, image_sets, how_many, category,
                                  jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        # 随机一个图片的编号加入当前的训练数据。
        image_index = random.randrange(65536)
        image_lists = image_sets[category]
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        bottleneck, ground_truth = get_or_create_bottleneck(sess, image_lists, image_index,
                                                            jpeg_data_tensor, bottleneck_tensor)
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


# 这个函数获取全部的测试数据。在最终测试的时候需要在所有的测试数据上计算正确率。
def get_test_bottlenecks(sess, image_sets, n_classes, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    image_lists = image_sets['testing']
    # 枚举所有的类别和每个类别中的测试图片。
    for index, _ in enumerate(image_lists):
        # 通过Inception-v3模型计算图片对应的特征向量，并将其加入最终数据的列表。
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        bottleneck, ground_truth = get_or_create_bottleneck(sess, image_lists, index,
                                                            jpeg_data_tensor, bottleneck_tensor)
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


def main(_):
    # 格式化数据集
    # format_data_set()
    # 读取所有图片。
    image_lists = create_image_sets()
    n_classes = 4
    # 读取已经训练好的Inception-v3模型。
    # 谷歌训练好的模型保存在了GraphDef Protocol Buffer中，里面保存了每一个节点取值的计算方法以及变量的取值。
    # TensorFlow模型持久化的问题在第5章中有详细的介绍。
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # 加载读取的Inception-v3模型，并返回数据输入所对应的张量以及计算瓶颈层结果所对应的张量。
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, return_elements=[BOTTLENECK_TENSOR_NAME,
                                                                                          JPEG_DATA_TENSOR_NAME])
    # 定义新的神经网络输入，这个输入就是新的图片经过Inception-v3模型前向传播到达瓶颈层时的结点取值。
    # 可以将这个过程类似的理解为一种特征提取。
    bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
    # 定义新的标准答案输入
    ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')
    # 定义一层全连接层来解决新的图片分类问题。
    # 因为训练好的Inception-v3模型已经将原始的图片抽象为了更加容易分类的特征向量了，所以不需要再训练那么复杂的神经网络来完成这个新的分类任务。
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input, weights) + biases
        final_tensor = tf.nn.softmax(logits)
    # 定义交叉熵损失函数
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)
    # 计算正确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        if os.path.exists(PARM_FILE + '.index'):
            # 从文件中恢复变量
            saver.restore(sess, PARM_FILE)
            print("Model restored.")
        else:
            print("Model inited.")
            tf.global_variables_initializer().run()
        sumloss = 0.0
        # 训练过程
        for i in range(STEPS):
            # 每次获取一个batch的训练数据
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
                sess, n_classes, image_lists, BATCH, 'training', jpeg_data_tensor, bottleneck_tensor)
            _, lossval = sess.run([train_step,cross_entropy_mean],
                     feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})
            sumloss = sumloss + lossval
            # 在验证集上测试正确率。
            if i % 100 == 0 or i + 1 == STEPS:
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(
                    sess, n_classes, image_lists, BATCH, 'validation', jpeg_data_tensor, bottleneck_tensor)
                validation_accuracy = sess.run(evaluation_step, feed_dict={
                    bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                print('Step %d: batch sum loss:%.2f Validation accuracy on random sampled %d examples = %.1f%%'
                      % (i, sumloss, BATCH, validation_accuracy * 100))
                sumloss = 0.0
                # 存储变量到文件
                if i % 1000 == 0:
                    save_path = saver.save(sess, PARM_FILE)
                    print("Model saved in file: ", save_path)

        # 在最后的测试数据上测试正确率
        test_bottlenecks, test_ground_truth = get_test_bottlenecks(sess, image_lists, n_classes,
                                                                   jpeg_data_tensor, bottleneck_tensor)
        test_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: test_bottlenecks,
                                                             ground_truth_input: test_ground_truth})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))


if __name__ == '__main__':
    tf.app.run()
