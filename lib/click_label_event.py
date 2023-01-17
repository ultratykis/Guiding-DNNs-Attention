import numpy as np
import cv2
from skimage.segmentation import felzenszwalb
from skimage.util import img_as_float
from pytorch_grad_cam.utils.image import show_cam_on_image

from lib.data import UnNormalize


class Candidate():
    def __init__(self, label_id, grad_cam, input_img, label, train, data_id):
        self.label_id = label_id
        self.un_norm = UnNormalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.image = show_cam_on_image(self.un_norm(
            input_img).numpy().transpose(1, 2, 0), cv2.resize(grad_cam, (224, 224,)))
        self.pos_region = []
        self.neg_region = []
        self.polygon_points = []
        self.label = label
        self.data_id = data_id
        if train:
            self.grad_cam = cv2.resize(grad_cam, (224, 224,))
            self.attention_peak = barycenter(self.grad_cam)
            self.attention_gt = self.attention_peak
            self.gcam_clusters = segment_gcam(self.grad_cam)
            self.candidate = {
                "label_id": label_id,
                "img": self.image,
                "grad_cam": grad_cam,
                "gcam_clusters": self.gcam_clusters
            }
        else:
            self.candidate = {
                "label_id": label_id,
                "img": self.image
            }

    def mouse_operation(self, mouse_operation, annotation_method):
        mouse_event = mouse_operation["click_type"]
        ev_pos_x = int(mouse_operation["click_position_x"])
        ev_pos_y = int(mouse_operation["click_position_y"])
        if mouse_event == "left_click" and annotation_method == "click":

            self.attention_gt = np.array([ev_pos_x, ev_pos_y])
            img = np.zeros((self.image.shape[0], self.image.shape[1]))
            pos = cv2.circle(img, (ev_pos_x, ev_pos_y), 5, 255, -1)
            for y in range(self.image.shape[0]):
                for x in range(self.image.shape[1]):
                    if pos[y][x] == 255:
                        self.pos_region.append((y, x))

            return self.updated_candidate(annotation_method)
        elif mouse_event == "left_click" and annotation_method == "polygon":
            self.polygon_points.append((ev_pos_x, ev_pos_y))
            img = np.zeros((self.image.shape[0], self.image.shape[1]))

            cv2.fillConvexPoly(img, np.array(
                self.polygon_points), color=255)
            for y in range(self.image.shape[0]):
                for x in range(self.image.shape[1]):
                    if img[y][x] == 255:
                        self.pos_region.append((y, x))

            vertices = np.array(self.polygon_points)
            self.attention_gt = np.array([vertices.mean(
                axis=0)[0], vertices.mean(axis=0)[1]])
            return self.updated_candidate(annotation_method)
        elif mouse_event == "right_click":
            # self.neg_region.clear()

            cluster_id = self.gcam_clusters[ev_pos_y, ev_pos_x]
            for y in range(self.gcam_clusters.shape[0]):
                for x in range(self.gcam_clusters.shape[1]):
                    if self.gcam_clusters[y, x] == cluster_id:
                        self.neg_region.append((y, x))

            return self.updated_candidate(annotation_method)
        else:
            self.pos_region.clear()
            self.neg_region.clear()
            self.polygon_points.clear()

            return self.image

    def updated_candidate(self, annotation_method):
        img_pos = np.zeros(self.image.shape)
        img_neg = np.zeros(self.image.shape)

        # for y, x in self.pos_region:
        #     img_pos[y][x] = [0, 0, 255]
        if len(self.pos_region) != 0 and annotation_method == 'click':
            for y, x in self.pos_region:
                # img_pos[y][x] = [0, 0, self.pos_region_full[y][x] * 255]
                img_pos[y][x] = [0, 0, 255]
        elif len(self.pos_region) != 0 and annotation_method == 'polygon':
            for y, x in self.pos_region:
                img_pos[y][x] = [0, 0, 255]

        if len(self.neg_region) != 0:
            for y, x in self.neg_region:
                img_neg[y][x] = [255, 0, 0]

        img_attention = cv2.addWeighted(img_pos, 1, img_neg, 1, 0)

        return cv2.addWeighted(
            self.image, 0.7, np.uint8(img_attention), 1, 0)


def barycenter(img):
    mu = cv2.moments(img*255, False)
    if mu['m00'] != 0:
        cx = int(mu['m10'] / mu['m00'])
        cy = int(mu['m01'] / mu['m00'])
    else:
        cx, cy = 0, 0
        Exception("No object detected")
    return np.array([cx, cy])


def segment_gcam(img):
    image = img_as_float(img)
    labels = felzenszwalb(image, scale=200, sigma=0.95, min_size=35)
    return labels


def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store
