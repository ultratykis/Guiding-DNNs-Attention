import base64
import os

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from pytorch_grad_cam import GradCAM
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import models

from lib.uncertainty import Uncertainty
from lib.click_label_event import *
from lib.data import ImageFolderWithMarkers
from lib.models.resnet import data_transform

app = Flask(__name__, static_folder='./dist/static',
            template_folder='./dist')
CORS(app)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

show_batch_size = 16
show_iter = 0
cycle = 0
active_learning = None
train_set = None
test_set = None
val_set = None
loader_iter = None
candidates = []
test_candidates = []
checkpoint_dir = './checkpoints'
train_batch_size = 128
added_number = 128
learnings_rate = 0.1
epoch = 100
weight_decay = 0.0005
dataset_path = ""
pretrained_network_file = ""
train = True
dataset = ""
selected_indices = []
candidates_sampling_strategy = "random"
experiment_id = None
class_labels = None
annotation_method = "click"


@app.route('/', defaults={'path': ''})
def index(path):
    return render_template('index.html')


@app.route('/start', methods=['post'])
def start():
    global experiment_id
    global learnings_rate
    global train_batch_size
    global epoch
    global weight_decay
    global dataset_path
    global pretrained_network_file
    global train
    global show_batch_size
    global candidates
    global dataset
    global candidates_sampling_strategy
    global time_point_start
    global annotation_method
    global cycle

    reset_params()
    experiment_id = request.json["experiment_id"]
    learnings_rate = float(request.json["learnings_rate"])
    train_batch_size = int(request.json["batch_size"])
    epoch = int(request.json["epoch"])
    weight_decay = float(request.json["weight_decay"])
    show_batch_size = int(request.json["show_batch_size"])
    dataset = request.json["dataset"]
    dataset_path = './datasets/{}/'.format(
        request.json["dataset"])
    pretrained_network_file = os.path.join(checkpoint_dir,
                                           request.json["dataset"], 'resnet18.pth')
    train = request.json["train"]
    candidates_sampling_strategy = request.json["candidates_sampling_strategy"]
    if request.json["annotation_method"]:
        annotation_method = "click"
    else:
        annotation_method = "polygon"

    reset_params()
    load_network()
    candidates_final = candidates_update()
    cycle = 0
    return jsonify(({"response_type": "started", "cycle": cycle,
                     "iter": show_iter,
                     "show_batch_size": show_batch_size, "candidates": candidates_final}))


@app.route('/test', methods=['post'])
def test():
    global active_learning
    global train
    global test_candidates
    test_candidates = []
    train = request.json["request_type"]
    cam = GradCAM(model=active_learning.model, target_layers=[
        active_learning.model.layer4[-1]], use_cuda=True)
    cam.batch_size = train_batch_size
    for idx, data in enumerate(active_learning.val_loader):
        inputs = data[0]
        labels = data[1]
        grad_cam = cam(input_tensor=inputs.cuda(), targets=None)
        set_test_image_labels(grad_cam, inputs, labels, idx, data[6])
    candidates_final = test_candidates_update()
    return jsonify(({"response_type": "test", "img_batch_size": len(candidates_final), "candidates": candidates_final}))


@app.route('/stop', methods=['post'])
def stop():
    reset_params()
    return jsonify(({"response_type": "stopped"}))


@app.route('/finetune', methods=['post'])
def finetune():
    global show_iter
    global cycle

    next_button_click()
    candidates_final = candidates_update()
    return jsonify(({"response_type": "started", "cycle": cycle,
                     "iter": show_iter,
                     "show_batch_size": show_batch_size, "candidates": candidates_final}))


@app.route('/annotation_style', methods=['post'])
def switch_annotation_style():
    global annotation_method
    if request.json["request_type"] == "annotation_style":
        annotation_method = request.json["annotation_method"]
        return jsonify(({"response_type": "annotation_style", "annotation_method": annotation_method}))
    else:
        return jsonify({"response_type": "unknown request"})


@app.route('/refine_cam', methods=['post'])
def refine_cam():
    global annotation_method
    img_id = request.json["img_id"]
    mouse_operation = request.json["mouse_operation"]
    refined_cam_img = candidates[img_id].mouse_operation(
        mouse_operation, annotation_method)
    img_base64 = base64.b64encode(refined_cam_img).decode('utf-8')
    return jsonify(({"response_type": "refine_cam", "refined_cam_img": {"cycle": cycle,
                                                                        "id": img_id,
                                                                        "img_src": "data:image/png;base64," + img_base64,
                                                                        "img_work_status": "done"}}))


def reset_params():
    global active_learning
    global train_set
    global test_set
    global loader_iter
    global show_iter
    global cycle
    show_iter = 0
    cycle = 0
    active_learning = None
    train_set = None
    test_set = None
    loader_iter = None
    torch.cuda.empty_cache()
    import gc
    gc.collect()


def load_network():
    global active_learning

    global train_set
    global test_set
    global val_set
    global class_labels

    train_set = ImageFolderWithMarkers(
        root=dataset_path+'train', transform=data_transform)
    uncertainty_train_set = ImageFolderWithMarkers(
        root=dataset_path+'train', transform=data_transform)
    test_set = ImageFolderWithMarkers(
        root=dataset_path+'test', transform=data_transform)
    val_set = ImageFolderWithMarkers(
        root=dataset_path+'val', transform=data_transform)
    print(train_set.class_to_idx)
    num_classes = len(train_set.class_to_idx)
    class_labels = train_set.classes
    network = models.resnet18(num_classes=num_classes)

    network.load_state_dict(torch.load(
        pretrained_network_file, map_location=torch.device('cuda'))['model_state_dict'])

    active_learning = Uncertainty(network.cuda(), added_number, train_set, uncertainty_train_set, test_set, val_set,
                                  train_batch_size, subset_number=512, sample_type=candidates_sampling_strategy, num_classes=num_classes)
    active_learning.set_hyperparams(
        epoch, lr=learnings_rate, milestones=[6, 12], wdecay=weight_decay)
    print("model loaded.")
    learn_guidance()


def learn_guidance():
    global cycle
    global show_iter
    global loader_iter
    global selected_indices
    global candidates
    global active_learning

    active_learning.update_train_loader()

    active_learning.learn(checkpoint_dir)
    selection_loader = DataLoader(active_learning.train_set, batch_size=show_batch_size,
                                  sampler=SubsetRandomSampler(
                                      active_learning.uncertainty_set_id),
                                  pin_memory=False)
    loader_iter = iter(selection_loader)
    selected_indices = selection_loader.sampler.indices
    candidates = []
    cycle += 1
    show_iter = 0
    show_next()


def show_next():
    global active_learning
    global selected_indices

    inputs, labels, _, _, _ = next(loader_iter)
    if show_iter == 0:
        indices = selected_indices[0:show_batch_size]
    else:
        indices = selected_indices[show_iter *
                                   show_batch_size:show_iter*show_batch_size+show_batch_size]
    cam = GradCAM(model=active_learning.model, target_layers=[
        active_learning.model.layer4[-1]], use_cuda=True)
    grad_cam = cam(input_tensor=inputs.cuda(), targets=None)
    print("cycle: " + str(cycle) + ' iter: ' + str(show_iter))
    set_image_labels(grad_cam, inputs, labels, indices)


def set_image_labels(grad_cams, inputs, labels, data_id):
    global candidates

    labels = labels.tolist()
    for idx in range(show_batch_size):
        label_id = show_iter * show_batch_size + idx
        label = class_labels[labels[idx]]
        candidate = Candidate(
            label_id, grad_cams[idx], inputs[idx], label, train, data_id[idx].tolist())
        candidates.append(candidate)


def set_test_image_labels(grad_cams, inputs, labels, iter, data_id):
    global test_candidates

    labels = labels.tolist()
    for idx in range(show_batch_size):
        label_id = iter * show_batch_size + idx
        label = class_labels[labels[idx]]
        candidate = Candidate(
            label_id, grad_cams[idx], inputs[idx], label, train, data_id)
        test_candidates.append(candidate)


def next_button_click():
    global show_batch_size
    global active_learning
    global show_iter
    if show_iter * show_batch_size == active_learning.added_number:
        get_guidance()
        print("Now start fine-tuning.")
        learn_guidance()
    else:
        get_guidance()
        print('Annotated!')
        show_next()


def get_guidance(random_points=False):
    global show_batch_size
    global candidates
    global show_iter
    global cycle
    global annotation_method
    global active_learning
    for i in range(show_batch_size):
        label_id = (show_iter - 1) * show_batch_size + i
        label_widget = candidates[label_id]
        attention_gt = label_widget.attention_gt
        attention_peak = label_widget.attention_peak
        nega_click = label_widget.grad_cam.copy()
        if random_points:
            attention_peak = np.random.randint(
                0, label_widget.grad_cam.shape[0], size=2)
            neg_region = get_random_neg(label_widget.gcam_clusters)
            if len(neg_region) != 0:
                for y, x in neg_region:
                    nega_click[y, x] = 0.0
        else:
            if len(label_widget.neg_region) != 0:
                for y, x in label_widget.neg_region:
                    nega_click[y, x] = 0.0
                mu = cv2.moments(nega_click*255, False)
                if mu['m00'] != 0:
                    attention_peak = np.array([
                        int(mu['m10'] / mu['m00']), int(mu['m01'] / mu['m00'])])
                else:
                    attention_peak = np.array([0, 0])

        path, target, neg_loss, refined_markers, attention_peak_temp = active_learning.train_set.samples[
            label_widget.data_id]

        neg_loss = (np.array(label_widget.grad_cam) -
                    np.array(nega_click)).mean()

        active_learning.train_set.samples[label_widget.data_id] = (
            path, target, neg_loss, attention_gt, attention_peak)  # set marker

    print("finished cycle:", cycle, "show iter:", show_iter)


def candidates_update():
    global show_iter
    candidates_final = []
    for idx in range(show_batch_size):
        label_id = show_iter * show_batch_size + idx
        img_base64 = base64.b64encode(
            candidates[label_id].image).decode('utf-8')
        candidates_final.append({"cycle": cycle,
                                 "id": label_id,
                                 "label": candidates[label_id].label,
                                 "img_src": "data:image/png;base64," + img_base64,
                                 "img_work_status": ""})

    show_iter += 1
    return candidates_final


def test_candidates_update():
    candidates_final = []
    for idx in range(len(test_candidates)):
        label_id = idx
        img_base64 = base64.b64encode(
            test_candidates[label_id].image).decode('utf-8')
        candidates_final.append({"cycle": cycle,
                                 "id": label_id,
                                 "label": test_candidates[label_id].label,
                                 "img_src": "data:image/png;base64," + img_base64,
                                 "img_work_status": ""})

    return candidates_final


def get_random_neg(img):
    width = img.shape[1]
    height = img.shape[0]
    x = np.random.randint(0, width)
    y = np.random.randint(0, height)
    cluster_id = img[y, x]
    neg_region = []
    for i in range(width):
        for j in range(height):
            if img[j, i] == cluster_id:
                neg_region.append((j, i))

    return neg_region


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
