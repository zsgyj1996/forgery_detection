import os
import cv2
import numpy as np
import sys
import argparse
import pdb
from sklearn import metrics
from tqdm import tqdm
import pickle

def read_annotations(data_path):
    data = []
    if os.path.exists(data_path):
        # 遍历指定目录下的所有文件
        for root, dirs, files in os.walk(data_path):
            for file in files:
                # 获取文件的绝对路径并添加到列表中
                file_path = os.path.abspath(os.path.join(root, file))
                sample_path = os.path.join(file_path.split('.')[0] + '.png').replace('val_gt', 'val_img')
                mask_path = file_path
                label = 1
            data.append((sample_path, mask_path, label))
    return data


def str2bool(in_str):
    if in_str in [1, "1", "t", "True", "true"]:
        return True
    elif in_str in [0, "0", "f", "False", "false", "none"]:
        return False


def calculate_img_score(pd, gt):
    seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
    true_pos = float(np.logical_and(pd, gt).sum())
    false_pos = np.logical_and(pd, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, gt).sum()
    true_neg = float(np.logical_and(seg_inv, gt_inv).sum())
    acc = (true_pos + true_neg) / (true_pos + true_neg + false_neg + false_pos + 1e-6)
    sen = true_pos / (true_pos + false_neg + 1e-6)
    spe = true_neg / (true_neg + false_pos + 1e-6)
    f1 = 2 * sen * spe / (sen + spe)
    return acc, sen, spe, f1, true_pos, true_neg, false_pos, false_neg


def calculate_pixel_f1(pd, gt):
    if np.max(pd) == np.max(gt) and np.max(pd) == 0:
        f1, iou = 1.0, 1.0
        return f1, 0.0, 0.0
    seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
    true_pos = float(np.logical_and(pd, gt).sum())
    false_pos = np.logical_and(pd, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, gt).sum()
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    precision = true_pos / (true_pos + false_pos + 1e-6)
    recall = true_pos / (true_pos + false_neg + 1e-6)
    return f1, precision, recall

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--pred_dir', type=str, default='save_out')
    parser.add_argument('--gt_file', type=str, default='None')
    parser.add_argument('--th', type=float, default=0.5)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = parse_args()
    annotation_file = opt.gt_file
    dataset = os.path.basename(annotation_file).split('.')[0]

    if not os.path.exists(annotation_file):
        print("%s not exists, quit" % annotation_file)
        sys.exit()
    annotation = read_annotations(annotation_file)
    scores, labs = [], []
    f1s = [[], [], []]

    results = []
    for ix, (img, mask, lab) in enumerate(tqdm(annotation)):
        pred_path = os.path.join(opt.pred_dir, os.path.basename(img).split('.')[0] + '_colored.png')
        try:
            pred = cv2.imread(pred_path, 0) / 255.0
        except:
            print("%s not exists" % pred_path)
            continue
        score = np.max(pred)
        scores.append(score)
        labs.append(lab)
        f1 = 0
        if lab != 0:
            try:
                gt = cv2.imread(mask, 0) / 255.0
            except:
                pdb.set_trace()
            if pred.shape != gt.shape:
                print("%s size not match" % pred_path)
                continue
            pred = (pred > opt.th).astype(np.float)
            try:
                f1, p, r = calculate_pixel_f1(pred.flatten(), gt.flatten())
            except:
                import pdb
                pdb.set_trace()
            f1s[lab-1].append(f1)

    fpr, tpr, thresholds = metrics.roc_curve((np.array(labs) > 0).astype(int), scores, pos_label=1)
    try:
        img_auc = metrics.roc_auc_score((np.array(labs) > 0).astype(int), scores)
    except:
        print("only one class")
        img_auc = 0.0

    meanf1 = np.mean(f1s[0]+f1s[1]+f1s[2])
    print("pixel-f1: %.4f" % meanf1)

    acc, sen, spe, f1_imglevel, tp, tn, fp, fn = calculate_img_score((np.array(scores) > 0.5), (np.array(labs) > 0).astype(int))
    print("img level acc: %.4f sen: %.4f  spe: %.4f  f1: %.4f auc: %.4f"
          % (acc, sen, spe, f1_imglevel, img_auc))
    print("combine f1: %.4f" % (2*meanf1*f1_imglevel/(f1_imglevel+meanf1+1e-6)))