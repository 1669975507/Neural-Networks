import numpy as np  #注意python和numpy版本的对应，如果不对应提示出错。

class Loader(object):

    def __init__(self, path, count):
        self.path = path
        self.count = count

    def get_file_content(self):
        print(self.path)
        f = open(self.path, 'rb')
        content = f.read()
        f.close()
        return content
    # def to_int(self, byte):
    #     return struct.unpack('B', byte)[0]

class ImageLoader(Loader):

    def get_picture(self, content, index):
        start = index * 28 * 28 + 16
        picture = []
        for i in range(28):
            picture.append([])
            for j in range(28):
                byte1 = content[start + i * 28 + j]
                picture[i].append(byte1)
        return picture

    def get_one_sample(self, picture):
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample

    def load(self,onerow=False):
        content = self.get_file_content()
        data_set = []
        for index in range(self.count):
            onepic =self.get_picture(content, index)
            if onerow: onepic = self.get_one_sample(onepic)
            data_set.append(onepic)
        return data_set

class LabelLoader(Loader):

    def load(self):
        content = self.get_file_content()
        labels = []
        for index in range(self.count):
            onelabel = content[index + 8]
            onelabelvec = self.norm(onelabel)
            labels.append(onelabelvec)
        return labels

    def norm(self, label):
        label_vec = []
        # label_value = self.to_int(label)
        label_value = label
        for i in range(10):
            if i == label_value:
                label_vec.append(1)
            else:
                label_vec.append(0)
        return label_vec

def get_training_data_set(num,onerow=False):
    image_loader = ImageLoader('D:/train-images.idx3-ubyte', num)
    label_loader = LabelLoader('D:/train-labels.idx1-ubyte', num)
    return image_loader.load(onerow), label_loader.load()

def get_test_data_set(num,onerow=False):
    image_loader = ImageLoader('D:/t10k-images.idx3-ubyte', num)
    label_loader = LabelLoader('D:/t10k-labels.idx1-ubyte', num)
    return image_loader.load(onerow), label_loader.load()

def printimg(onepic):
    onepic=onepic.reshape(28,28)
    for i in range(28):
        for j in range(28):
            if onepic[i, j] == 0:
                print('   ', end='')  # python3支持这种格式，如果是python2不支持print这种格式
            else:  # print "* "
                print('* ', end='')  # python3geshi
        print('')


if __name__=="__main__":
    train_data_set, train_labels = get_training_data_set(100)
    train_data_set = np.array(train_data_set)
    train_labels = np.array(train_labels)
    for i in range(28): #实现输出前28个手写体数字。

        onepic = train_data_set[i]
        printimg(onepic)
        print(train_labels[i].argmax())