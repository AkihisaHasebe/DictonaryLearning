import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.metrics import roc_auc_score, roc_curve
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import timm
from tqdm import tqdm
from data import MVTecDataset
import argparse
from pathlib import Path
import cv2


class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        return img

class Extract_feature_map():
    def __init__(self, model_name='resnet50', pool_last=False) -> None:
        
        # ResNet50のプーリング層の位置を指定
        self.layer_indices = (1, 2, 3)
        self.model = timm.create_model(model_name, pretrained=True,features_only=True,out_indices=self.layer_indices)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.pool = torch.nn.AdaptiveAvgPool2d(1) if pool_last else None
        self.backbone_name = model_name
        self.devide = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.devide)


    def extract_feature_map(self, image):

        with torch.no_grad():
            feature_maps = self.model(image.to(self.devide))

        feature_maps = [fmap.to('cpu') for fmap in feature_maps]
        
        largest_fmap_size = feature_maps[0].shape[-2:]
        resize = torch.nn.AdaptiveAvgPool2d(largest_fmap_size)
        average = torch.nn.AvgPool2d(3,stride=1)
        resized_maps = [resize(average(fmap)) for fmap in feature_maps]
        patch = torch.cat(resized_maps, 1)
        patch = patch.permute(0,2,3,1)
        patch = patch.reshape(-1, patch.shape[3])

        return patch



def save_anomaly_map(original_img, anomaly_map, output_dir, alpha=0.3):
    colored_anomaly_map = cv2.applyColorMap((((anomaly_map - np.min(anomaly_map))/(np.max(anomaly_map) - np.min(anomaly_map))) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    # colored_anomaly_map = cv2.resize(colored_anomaly_map,original_img.size,interpolation=cv2.INTER_LINEAR)
    blended_image = cv2.addWeighted(cv2.cvtColor(np.array(original_img),cv2.COLOR_RGB2BGR), 1.0-alpha, colored_anomaly_map, alpha , 0)
    blended_image = cv2.cvtColor(blended_image,cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1,3,dpi=144,figsize=(12,8))

    ax[0].imshow(original_img)
    ax[1].imshow(anomaly_map)
    ax[2].imshow(blended_image)

    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    
    fig.savefig(output_dir)
    plt.close(fig)

    np.save(Path(output_dir).with_suffix('.npy'),anomaly_map)


def main(arg_dict:dict):
    # 学習フェイズ
    cls = arg_dict['class']

    extracter = Extract_feature_map(model_name=arg_dict['backbone'])

    batch_size = arg_dict['batch_size']

    if arg_dict['pre_resize'] is None:
        pre_resize = arg_dict['size']

    elif arg_dict['pre_resize'] > arg_dict['size']:
        pre_resize = arg_dict['pre_resize']
    else:
        raise ValueError('"--pre_resizse" should be larger than "--size". But now "--pre_resize" is {} and "--size" is {}'\
                         .format(arg_dict['pre_resize'], arg_dict['size']))

    dataset = MVTecDataset(cls=cls,size=arg_dict['size'],pre_resize=pre_resize)
    train_dataset, val_dataset = dataset.get_datasets()

    # 実行した日時を名前としたディレクトリを作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path('res')
    output_dir = output_dir.joinpath(timestamp,cls)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    if arg_dict['dictionary_path'] is None:

        dictionary = MiniBatchDictionaryLearning(n_components=arg_dict['n_components'], alpha=1.0,\
                                                max_iter=1,transform_algorithm='lasso_lars',\
                                                n_jobs=arg_dict['n_jobs'],batch_size=arg_dict['dict_batchsize'],
                                                random_state=arg_dict['seed'])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(arg_dict['epoch']):
            loss_epoch = 0
            for images,_ in tqdm(train_dataloader,total=len(train_dataloader),leave=False,desc="Epoch {}".format(epoch)):
                feature_map = extracter.extract_feature_map(images)
                feature_map_np = feature_map.detach().numpy().astype(np.float64)
                dictionary.partial_fit(feature_map_np)

                sparse_codes = dictionary.transform(feature_map_np)
                reconstructed_patches = np.dot(sparse_codes, dictionary.components_)

                loss = np.mean(np.sqrt((feature_map_np - reconstructed_patches)**2))
                loss_epoch += loss

            print('Epoch {} Loss: {}'.format(epoch, loss_epoch/len(train_dataloader)))

        joblib.dump(dictionary, output_dir.joinpath('dictionary_model.pkl'))

    else:
        dictionary = joblib.load(arg_dict['dictionary_path'])


    # 検証フェイズ
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    image_preds = []
    pixel_preds = []
    image_labels = []
    pixel_labels = []

    for i, (images, mask, label) in tqdm(enumerate(val_dataloader),total=len(val_dataloader)):
        # 特徴マップの抽出
        feature_map = extracter.extract_feature_map(images)
        feature_map_np = feature_map.detach().numpy().astype(np.float64)

        # パッチ抽出
        # patches = extract_patches_2d(feature_map_np, patch_size)

        # スパース表現
        sparse_codes = dictionary.transform(feature_map_np)

        # 再構成
        reconstructed_patches = np.dot(sparse_codes, dictionary.components_)

        anomaly_vec = np.mean(np.sqrt((feature_map_np - reconstructed_patches)**2),axis=1)

        anomaly_map = anomaly_vec.reshape(arg_dict['size']//4,arg_dict['size']//4)
        anomaly_map = cv2.resize(anomaly_map,(arg_dict['size'],arg_dict['size']),cv2.INTER_LINEAR)



        image_preds.append(np.max(anomaly_map))
        pixel_preds.extend(anomaly_map.flatten())
        
        image_labels.append(label)
        pixel_labels.extend(mask.flatten().numpy())


        # 誤差マップの保存
        org_imgname = Path(val_dataset.imgs[i][0])
        img = Image.open(org_imgname)
        img = img.resize((pre_resize, pre_resize))
        crop = transforms.CenterCrop(arg_dict['size'])
        img = crop.forward(img)
        # save_error_map(img,feature_map_np, reconstructed_feature_map,os.path.join(output_dir,f"error_map_{i * batch_size + j}.png"))
        save_anomaly_map(img, anomaly_map, os.path.join(output_dir,f'{org_imgname.parts[-2]}_{org_imgname.name}'),alpha=0.3)

    image_labels = np.stack(image_labels)
    image_preds = np.stack(image_preds)

    image_rocauc = roc_auc_score(image_labels, image_preds)
    pixel_rocauc = roc_auc_score(pixel_labels, pixel_preds)

    print('Image ROC AUC:',image_rocauc)
    print('Pixel ROC AUC', pixel_rocauc)


def parse_arguments():
    parser = argparse.ArgumentParser(description="コマンドライン引数を取り込むプログラム")
    parser.add_argument("--class", type=str, required=True, help="クラスの名前を指定してください")
    parser.add_argument("--size", type=int, default=224,  help="クラスの名前を指定してください")
    parser.add_argument("--pre_resize", type=int, default=None, help="前処理Centercrop前のResizeの出力サイズ. Noneでは--sizeの1.142倍で処理される")
    parser.add_argument("--epoch", type=int, default=50,  help="クラスの名前を指定してください")
    parser.add_argument("--batch_size", type=int, default=50,  help="クラスの名前を指定してください")
    parser.add_argument("--n_jobs", type=int, default=None, help="MiniBatchDictionaryLearningで使用するCPUコア数の指定.Noneでは1つのコアを使って計算する.")
    parser.add_argument("--dictionary_path", type=Path, default=None, help="学習済み辞書のパスを指定．指定された場合はその辞書を読み込む")
    parser.add_argument("--dict_batchsize", type=int, default=256,help="辞書学習内のバッチサイズ")
    parser.add_argument("--seed", type=int, default=42, help="randam seed")
    parser.add_argument("--backbone", type=str, default='resnet18', help='backbone model name')
    parser.add_argument("--n_components",type=int, default=100)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    arg_dict = vars(args)
    main(arg_dict)