import torch.utils.data as data 导入PyTorch的utils.data 模块，并简称为data
import json 导入Python的json模块,用于处理JSON格式的数据（读取和写入JSON文件）。
import os 导入Python的os模块，用于与操作系统交互，比如文件路径操作、目录遍历等。
import subprocess 导入Python的subprocess模块，用于创建子进程并执行外部命令（比如调用系统命令或运行其他程序）。
from PIL import Image 从Python Imaging Library（Pillow库）导入Image模块，用于图像处理（如打开、保存、调整图像等）。
import numpy as np 导入NumPy库，并简称为np。NumPy是Python中用于科学计算的核心库，提供多维数组和矩阵运算功能。
import torch 导入PyTorch库，这是一个广泛使用的深度学习框架，支持张量计算、自动微分和GPU加速。
import pickle 导入Python的pickle模块，用于对象的序列化和反序列化（将Python对象保存为文件或从文件加载对象）。
from util import * 从util.py 文件中导入所有内容（*表示导入所有函数、类或变量）。

urls = {'train_img':'http://images.cocodataset.org/zips/train2014.zip',
        'val_img' : 'http://images.cocodataset.org/zips/val2014.zip',
        'annotations':'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'}

def download_coco2014(root, phase): root: 数据集存储的根目录（如 ./data/coco）。phase: 指定下载的数据类型，可以是 'train'（训练集）或 'val'（验证集）。

    if not os.path.exists(root):  #如果根目录 root 不存在，则创建它。
        os.makedirs(root)
    tmpdir = os.path.join(root, 'tmp/') # 临时下载目录
    data = os.path.join(root, 'data/') # 最终数据存储目录
    
    if not os.path.exists(data):
        os.makedirs(data)
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)  #确保 data/ 和 tmp/ 目录存在。
        
    if phase == 'train':
        filename = 'train2014.zip'
    elif phase == 'val':
        filename = 'val2014.zip' #根据 phase 选择下载 train2014.zip 或 val2014.zip。
        
    cached_file = os.path.join(tmpdir, filename) #cached_file 是 ZIP 文件的完整路径（如 ./data/coco/tmp/train2014.zip）
    
    if not os.path.exists(cached_file):
        print('Downloading: "{}" to {}\n'.format(urls[phase + '_img'], cached_file)) #如果文件不存在，打印下载信息
        
        os.chdir(tmpdir) #切换到临时目录 tmpdir
        subprocess.call('wget ' + urls[phase + '_img'], shell=True) #通过 wget 下载 COCO 数据集文件。
        os.chdir(root) #切换回根目录 root
        
    # extract file
    img_data = os.path.join(data, filename.split('.')[0]) #img_data 是解压后的目录路径（如 ./data/coco/data/train2014）
    
    if not os.path.exists(img_data):
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=data))#使用字符串格式化（.format()）输出解压的源文件和目标路径
        command = 'unzip {} -d {}'.format(cached_file,data) #如果解压目录不存在，调用 unzip 解压文件到 data/
        os.system(command) #os.system 执行解压命令
    print('[dataset] Done!') #提示图像数据下载和解压完成。

    # train/val images/annotations
    cached_file = os.path.join(tmpdir, 'annotations_trainval2014.zip')
    if not os.path.exists(cached_file):
        print('Downloading: "{}" to {}\n'.format(urls['annotations'], cached_file))
        os.chdir(tmpdir)
        subprocess.Popen('wget ' + urls['annotations'], shell=True)
        os.chdir(root) 
        #检查目标文件是否已存在（annotations_trainval2014.zip），如果不存在则下载。
         使用 wget 命令下载文件，并保存到临时目录 tmpdir 中。
         切换工作目录（os.chdir）以确保文件下载到正确位置，最后恢复原始目录。
        
    annotations_data = os.path.join(data, 'annotations')
    if not os.path.exists(annotations_data):
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=data))
        command = 'unzip {} -d {}'.format(cached_file, data)
        os.system(command)
    print('[annotation] Done!')

    anno = os.path.join(data, '{}_anno.json'.format(phase))
    img_id = {}
    annotations_id = {}
    if not os.path.exists(anno):
        annotations_file = json.load(open(os.path.join(annotations_data, 'instances_{}2014.json'.format(phase))))
        annotations = annotations_file['annotations']
        category = annotations_file['categories']
        category_id = {}
        for cat in category:
            category_id[cat['id']] = cat['name']
        cat2idx = categoty_to_idx(sorted(category_id.values()))
        images = annotations_file['images']
        for annotation in annotations:
            if annotation['image_id'] not in annotations_id:
                annotations_id[annotation['image_id']] = set()
            annotations_id[annotation['image_id']].add(cat2idx[category_id[annotation['category_id']]])
        for img in images:
            if img['id'] not in annotations_id:
                continue
            if img['id'] not in img_id:
                img_id[img['id']] = {}
            img_id[img['id']]['file_name'] = img['file_name']
            img_id[img['id']]['labels'] = list(annotations_id[img['id']])
        anno_list = []
        for k, v in img_id.items():
            anno_list.append(v)
        json.dump(anno_list, open(anno, 'w'))
        if not os.path.exists(os.path.join(data, 'category.json')):
            json.dump(cat2idx, open(os.path.join(data, 'category.json'), 'w'))
        del img_id
        del anno_list
        del images
        del annotations_id
        del annotations
        del category
        del category_id
    print('[json] Done!')

def categoty_to_idx(category):
    cat2idx = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx


class COCO2014(data.Dataset):
    def __init__(self, root, transform=None, phase='train', inp_name=None):
        self.root = root
        self.phase = phase
        self.img_list = []
        self.transform = transform
        download_coco2014(root, phase)
        self.get_anno()
        self.num_classes = len(self.cat2idx)

        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name

    def get_anno(self):
        list_path = os.path.join(self.root, 'data', '{}_anno.json'.format(self.phase))
        self.img_list = json.load(open(list_path, 'r'))
        self.cat2idx = json.load(open(os.path.join(self.root, 'data', 'category.json'), 'r'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        return self.get(item)

    def get(self, item):
        filename = item['file_name']
        labels = sorted(item['labels'])
        img = Image.open(os.path.join(self.root, 'data', '{}2014'.format(self.phase), filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = np.zeros(self.num_classes, np.float32) - 1
        target[labels] = 1
        return (img, filename, self.inp), target
