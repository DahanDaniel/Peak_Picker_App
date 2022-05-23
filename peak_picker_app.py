import math
import os

import numpy as np
import scipy
import nmrglue as ng
import tensorflow as tf
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import *
from PyQt5 import QtCore

EXP_DATA_PATH = "azaron_surowe_dane"

N_SERIES = 20
N_PIXELS = 256
RESOLUTION = 256
PIXMAP_PIXELS = 400

# Constants from original script.
sp_min, sp_max = 10.0, -10.0
sp_pixels = 256
fr_pixels = 256
hf_min, hf_max = 1.0 / 16.0, 16.0
# Number of cells along the speed and frequency dimensions
sp_cells, fr_cells = 256, 256
# Offsets of cells relative to the entire grid in range from 0 to number of cells
fr_offset, sp_offset = tf.meshgrid(
    tf.range(fr_cells, dtype=tf.float32), tf.range(sp_cells, dtype=tf.float32)
)


def getxaxis(dic, data):
    dic2 = ng.varian.guess_udic(dic, data)
    dic2[0]["size"] = int(dic["procpar"]["np"]["values"][0]) / 2
    dic2[0]["complex"] = True
    dic2[0]["encoding"] = "direct"
    dic2[0]["sw"] = float(dic["procpar"]["sw"]["values"][0])
    dic2[0]["obs"] = float(dic["procpar"]["sfrq"]["values"][0])
    dic2[0]["car"] = (
        float(dic["procpar"]["sfrq"]["values"][0])
        - float(dic["procpar"]["reffrq"]["values"][0])
    ) * 1e6
    dic2[0]["label"] = dic["procpar"]["tn"]["values"][0]
    C = ng.convert.converter()
    C.from_varian(dic, data, dic2)
    pdic, pdata = C.to_pipe()
    A = ng.pipe.make_uc(pdic, pdata)
    B = A.ppm_limits()
    xaxis = np.linspace(B[0], B[1], data.shape[0])
    return xaxis


def read_fid_and_scale():
    data = np.empty((N_PIXELS * 128, N_SERIES), dtype=np.complex128)
    for i in range(20):
        dic, data[:, i] = ng.varian.read(
            os.path.join(EXP_DATA_PATH, str(i + 1) + ".fid")
        )
    fid = data * np.exp(-1j * np.angle(data[0]))
    fid[0] /= 2.0
    fid = tf.constant(fid, dtype=tf.complex64) / 1000000.0

    ppm_scale = getxaxis(dic, data[:, 0])

    return fid, ppm_scale


def radon_transform(fid):
    # Ranges in experiment.
    tm_begin, tm_end, tm_pixels = 0, 1, 256 * 128
    sp_begin, sp_end, sp_pixels = -10, 10, 256
    sr_begin, sr_end, sr_pixels = 0, 20, 20

    # Time, speed and series coordinates in experiment.
    tm_coord = tf.range(tm_begin, tm_end, tm_end / tm_pixels)
    sp_coord = tf.range(sp_begin, sp_end, (sp_end - sp_begin) / sp_pixels)
    sr_coord = tf.range(sr_begin, sr_end, (sr_end - sr_begin) / sr_pixels)

    # Shifting phase in experiment.
    phase = tf.exp(
        -2.0
        * math.pi
        * tf.complex(
            0.0, sp_coord[:, None, None] * tm_coord[:, None] * sr_coord
        )
    )

    # Radon spectrum from experiment.
    radon = phase * fid
    radon = tf.reduce_mean(radon, 2)
    radon = tf.signal.fft(radon)
    radon = tf.signal.fftshift(radon, axes=-1)
    radon = tf.math.real(radon)

    # three singlets are in this interval: [:, 20775: 20775 + 256]

    return radon


def get_detector_model():
    radon = tf.keras.Input([256, 256])
    signal1 = tf.keras.layers.Reshape([256, 256, 1])(radon)
    signal1 = tf.keras.layers.BatchNormalization()(signal1)
    signal1 = tf.keras.layers.Conv2D(32, 3, 1, "same", activation="relu")(
        signal1
    )
    signal1 = tf.keras.layers.Conv2D(32, 3, 1, "same", activation="relu")(
        signal1
    )
    signal2 = tf.keras.layers.MaxPool2D()(signal1)
    signal2 = tf.keras.layers.BatchNormalization()(signal2)
    signal2 = tf.keras.layers.Conv2D(64, 3, 1, "same", activation="relu")(
        signal2
    )
    signal2 = tf.keras.layers.Conv2D(64, 3, 1, "same", activation="relu")(
        signal2
    )
    signal3 = tf.keras.layers.MaxPool2D()(signal2)
    signal3 = tf.keras.layers.BatchNormalization()(signal3)
    signal3 = tf.keras.layers.Conv2D(64, 3, 1, "same", activation="relu")(
        signal3
    )
    signal3 = tf.keras.layers.Conv2D(64, 3, 1, "same", activation="relu")(
        signal3
    )
    signal3 = tf.keras.layers.UpSampling2D()(signal3)
    signal4 = tf.keras.layers.Concatenate()([signal2, signal3])
    signal4 = tf.keras.layers.BatchNormalization()(signal4)
    signal4 = tf.keras.layers.Conv2D(32, 3, 1, "same", activation="relu")(
        signal4
    )
    signal4 = tf.keras.layers.Conv2D(32, 3, 1, "same", activation="relu")(
        signal4
    )
    signal4 = tf.keras.layers.UpSampling2D()(signal4)
    signal5 = tf.keras.layers.Concatenate()([signal1, signal4])
    signal5 = tf.keras.layers.BatchNormalization()(signal5)
    signal5 = tf.keras.layers.Conv2D(16, 3, 1, "same", activation="relu")(
        signal5
    )
    signal5 = tf.keras.layers.Conv2D(16, 3, 1, "same", activation="relu")(
        signal5
    )
    signal5 = tf.keras.layers.Conv2D(1, 1, 1)(signal5)
    ob_logit = tf.keras.layers.Reshape([256, 256])(signal5)
    model = tf.keras.models.Model(radon, ob_logit)

    model.summary()

    # Load model weights.
    model.load_weights(r"Picker\07extract.h5")

    return model


def get_radon_slice(radon, slice_start):
    SLICE_WIDTH = 256
    max_idx = radon.shape[1] - 1
    low_bound = max(0, slice_start)
    upp_bound = min(max_idx, slice_start + SLICE_WIDTH)
    return radon[:, low_bound:upp_bound]


app = QApplication([])


class Picker(QWidget):
    def __init__(self, radon, model, ppm_scale, start_slice_idx=0):
        super().__init__()

        self.radon = radon
        self.model = model
        self.ppm_scale = ppm_scale

        self.fr_margin_pixel = start_slice_idx
        self.scale = 100.0
        self.add = 0.0

        self.sp_marker_min = 0
        self.sp_marker_max = 256
        self.fr_marker_min = 0
        self.fr_marker_max = 256

        self.colormap = plt.get_cmap()

        self.speedEditUnit = QLineEdit()
        self.frequencyEditUnit = QLineEdit()
        self.speedEditPts = QLineEdit()
        self.frequencyEditPts = QLineEdit()

        self.speedEditUnit.setFocusPolicy(Qt.NoFocus)
        self.frequencyEditUnit.setFocusPolicy(Qt.NoFocus)
        self.speedEditPts.setFocusPolicy(Qt.NoFocus)
        self.frequencyEditPts.setFocusPolicy(Qt.NoFocus)

        self.speedEditUnit.setFixedWidth(76)
        self.frequencyEditUnit.setFixedWidth(76)
        self.speedEditPts.setFixedWidth(76)
        self.frequencyEditPts.setFixedWidth(76)

        # Result layout.
        resultLayout = QGridLayout()
        resultLayout.addWidget(QLabel("Results:"), 0, 0)
        resultLayout.addWidget(QLabel("Speed:"), 2, 0)
        resultLayout.addWidget(self.speedEditUnit, 3, 0)
        resultLayout.addWidget(QLabel("ppb/K"), 3, 1)
        resultLayout.addWidget(self.speedEditPts, 4, 0)
        resultLayout.addWidget(QLabel("pts/series"), 4, 1)
        resultLayout.addWidget(QLabel("Freqency:"), 6, 0)
        resultLayout.addWidget(self.frequencyEditUnit, 7, 0)
        resultLayout.addWidget(QLabel("ppm"), 7, 1)
        resultLayout.addWidget(self.frequencyEditPts, 8, 0)
        resultLayout.addWidget(QLabel("pts"), 8, 1)

        resultPanel = QVBoxLayout()
        resultPanel.addStretch()
        resultPanel.addLayout(resultLayout)
        resultPanel.addStretch()

        self.subtractEdit = QLineEdit()
        self.divideEdit = QLineEdit()

        self.subtractEdit.editingFinished.connect(self.onSubtractEdited)
        self.divideEdit.editingFinished.connect(self.onDivideEdited)

        self.subtractEdit.setFixedWidth(76)
        self.divideEdit.setFixedWidth(76)

        verticalLayout = QGridLayout()
        verticalLayout.addWidget(QLabel("Sub"), 0, 0)
        verticalLayout.addWidget(self.subtractEdit, 0, 1)
        verticalLayout.addWidget(QLabel("Div"), 0, 2)
        verticalLayout.addWidget(self.divideEdit, 0, 3)

        # Edits in unit.
        self.speedMarkerUnitMinEdit = QLineEdit()
        self.frequencyRangeUnitMinEdit = QLineEdit()
        self.frequencyMarkerUnitMinEdit = QLineEdit()
        self.frequencyMarkerUnitMaxEdit = QLineEdit()
        self.frequencyRangeUnitMaxEdit = QLineEdit()
        self.speedMarkerUnitMaxEdit = QLineEdit()
        self.speedMarkerUnitMinEdit.editingFinished.connect(
            self.onSpeedMarkerUnitMinEdited
        )
        self.frequencyRangeUnitMinEdit.editingFinished.connect(
            self.onFrequencyRangeUnitMinEdited
        )
        self.frequencyMarkerUnitMinEdit.editingFinished.connect(
            self.onFrequencyMarkerUnitMinEdited
        )
        self.frequencyMarkerUnitMaxEdit.editingFinished.connect(
            self.onFrequencyMarkerUnitMaxEdited
        )
        self.frequencyRangeUnitMaxEdit.editingFinished.connect(
            self.onFrequencyRangeUnitMaxEdited
        )
        self.speedMarkerUnitMaxEdit.editingFinished.connect(
            self.onSpeedMarkerUnitMaxEdited
        )
        self.speedMarkerUnitMinEdit.setFixedWidth(100)
        self.frequencyRangeUnitMinEdit.setFixedWidth(100)
        self.frequencyMarkerUnitMinEdit.setFixedWidth(100)
        self.frequencyMarkerUnitMaxEdit.setFixedWidth(100)
        self.frequencyRangeUnitMaxEdit.setFixedWidth(100)
        self.speedMarkerUnitMaxEdit.setFixedWidth(100)

        # Edits in points.
        self.speedMarkerPtsMinEdit = QLineEdit()
        self.frequencyRangePtsMinEdit = QLineEdit()
        self.frequencyMarkerPtsMinEdit = QLineEdit()
        self.frequencyMarkerPtsMaxEdit = QLineEdit()
        self.frequencyRangePtsMaxEdit = QLineEdit()
        self.speedMarkerPtsMaxEdit = QLineEdit()
        self.speedMarkerPtsMinEdit.editingFinished.connect(
            self.onSpeedMarkerPtsMinEdited
        )
        self.frequencyRangePtsMinEdit.editingFinished.connect(
            self.onFrequencyRangePtsMinEdited
        )
        self.frequencyMarkerPtsMinEdit.editingFinished.connect(
            self.onFrequencyMarkerPtsMinEdited
        )
        self.frequencyMarkerPtsMaxEdit.editingFinished.connect(
            self.onFrequencyMarkerPtsMaxEdited
        )
        self.frequencyRangePtsMaxEdit.editingFinished.connect(
            self.onFrequencyRangePtsMaxEdited
        )
        self.speedMarkerPtsMaxEdit.editingFinished.connect(
            self.onSpeedMarkerPtsMaxEdited
        )
        self.speedMarkerPtsMinEdit.setFixedWidth(100)
        self.frequencyRangePtsMinEdit.setFixedWidth(100)
        self.frequencyMarkerPtsMinEdit.setFixedWidth(100)
        self.frequencyMarkerPtsMaxEdit.setFixedWidth(100)
        self.frequencyRangePtsMaxEdit.setFixedWidth(100)
        self.speedMarkerPtsMaxEdit.setFixedWidth(100)

        # Layout.
        horizontalLayout = QGridLayout()
        horizontalLayout.addWidget(QLabel("Frequency range:"), 0, 0)
        horizontalLayout.addWidget(QLabel("Left boundary:"), 1, 0)
        horizontalLayout.addWidget(self.frequencyRangeUnitMinEdit, 2, 0)
        horizontalLayout.addWidget(QLabel("ppm"), 2, 1)
        horizontalLayout.addWidget(self.frequencyRangePtsMinEdit, 3, 0)
        horizontalLayout.addWidget(QLabel("pts"), 3, 1)
        horizontalLayout.addWidget(QLabel("Right boundary:"), 1, 2)
        horizontalLayout.addWidget(self.frequencyRangeUnitMaxEdit, 2, 2)
        horizontalLayout.addWidget(QLabel("ppm"), 2, 3)
        horizontalLayout.addWidget(self.frequencyRangePtsMaxEdit, 3, 2)
        horizontalLayout.addWidget(QLabel("pts"), 3, 3)

        horizontalLayout.addWidget(QLabel("Frequency markers:"), 4, 0)
        horizontalLayout.addWidget(QLabel("Left boundary:"), 5, 0)
        horizontalLayout.addWidget(self.frequencyMarkerUnitMinEdit, 6, 0)
        horizontalLayout.addWidget(QLabel("ppm"), 6, 1)
        horizontalLayout.addWidget(self.frequencyMarkerPtsMinEdit, 7, 0)
        horizontalLayout.addWidget(QLabel("pts"), 7, 1)
        horizontalLayout.addWidget(QLabel("Right boundary:"), 5, 2)
        horizontalLayout.addWidget(self.frequencyMarkerUnitMaxEdit, 6, 2)
        horizontalLayout.addWidget(QLabel("ppm"), 6, 3)
        horizontalLayout.addWidget(self.frequencyMarkerPtsMaxEdit, 7, 2)
        horizontalLayout.addWidget(QLabel("pts"), 7, 3)

        horizontalLayout.addWidget(QLabel("Speed markers:"), 8, 0)
        horizontalLayout.addWidget(QLabel("Lower boundary:"), 9, 0)
        horizontalLayout.addWidget(self.speedMarkerUnitMinEdit, 10, 0)
        horizontalLayout.addWidget(QLabel("ppb/K"), 10, 1)
        horizontalLayout.addWidget(self.speedMarkerPtsMinEdit, 11, 0)
        horizontalLayout.addWidget(QLabel("pts"), 11, 1)
        horizontalLayout.addWidget(QLabel("Upper boundary:"), 9, 2)
        horizontalLayout.addWidget(self.speedMarkerUnitMaxEdit, 10, 2)
        horizontalLayout.addWidget(QLabel("ppb/K"), 10, 3)
        horizontalLayout.addWidget(self.speedMarkerPtsMaxEdit, 11, 2)
        horizontalLayout.addWidget(QLabel("pts"), 11, 3)

        infoLayout = QVBoxLayout()
        infoLayout.addWidget(QLabel("Options:"))
        infoLayout.addLayout(verticalLayout)
        infoLayout.addSpacing(16)
        infoLayout.addLayout(horizontalLayout)
        infoLayout.addStretch()

        controlsLayout = QVBoxLayout()
        controlsLayout.addWidget(QLabel("Controls:\n"))
        controlsLayout.addWidget(QLabel("<-: move left on the Radon spectrum"))
        controlsLayout.addWidget(QLabel("->: move right on the Radon spectrum"))
        controlsLayout.addWidget(QLabel("Ctrl + <-: narrow the frequency markers"))
        controlsLayout.addWidget(QLabel("Ctrl + ->: broaden the frequency markers"))
        controlsLayout.addWidget(QLabel("Ctrl + Down: narrow the speed markers"))
        controlsLayout.addWidget(QLabel("Ctrl + Up: broaden the speed markers"))
        controlsLayout.addStretch()
        # controlsLayout.setStyleSheet(""" font-size: 12px; """)

        self.rd_widget = QLabel()
        self.fr_widget = QLabel()
        self.sp_widget = QLabel()

        centerLayout = QGridLayout()
        centerLayout.addLayout(controlsLayout, 0, 0)
        centerLayout.addWidget(self.fr_widget, 0, 1)
        centerLayout.addWidget(self.sp_widget, 1, 0)
        centerLayout.addWidget(self.rd_widget, 1, 1)

        mainLayout = QGridLayout()
        mainLayout.addLayout(infoLayout, 0, 0)
        mainLayout.addLayout(centerLayout, 0, 1)
        mainLayout.addLayout(resultPanel, 0, 2)

        self.setLayout(mainLayout)

        self.detect()
        self.draw()
        self.setFocus()

    def convert_pts_to_ppm(self, pts):
        ppm = np.interp(pts, np.arange(len(self.ppm_scale)), self.ppm_scale)
        return ppm

    def convert_ppm_to_pts(self, ppm):
        pts = np.interp(
            ppm,
            np.flip(self.ppm_scale),
            np.linspace(len(self.ppm_scale) - 1, 0, len(self.ppm_scale)),
        )
        return pts

    def onSubtractEdited(self):
        self.add = float(self.subtractEdit.text())
        self.detect()
        self.draw()
        self.setFocus()

    def onDivideEdited(self):
        self.scale = float(self.divideEdit.text())
        self.detect()
        self.draw()
        self.setFocus()

    def onSpeedMarkerUnitMinEdited(self):
        sp_marker_min_param = self.convert_ppm_to_pts(
            float(self.speedMarkerUnitMinEdit.text())
        )
        sp_marker_min_unit = (sp_marker_min_param - sp_min) / (sp_max - sp_min)
        sp_marker_min_pixel = sp_marker_min_unit * sp_pixels
        self.sp_marker_min = round(sp_marker_min_pixel)
        self.draw()
        self.setFocus()

    def onSpeedMarkerUnitMaxEdited(self):
        sp_marker_max_param = self.convert_ppm_to_pts(
            float(self.speedMarkerUnitMaxEdit.text())
        )
        sp_marker_max_unit = (sp_marker_max_param - sp_min) / (sp_max - sp_min)
        sp_marker_max_pixel = sp_marker_max_unit * sp_pixels
        self.sp_marker_max = round(sp_marker_max_pixel)
        self.draw()
        self.setFocus()

    def onFrequencyMarkerUnitMinEdited(self):
        self.fr_marker_min = int(
            self.convert_ppm_to_pts(
                float(self.frequencyMarkerUnitMinEdit.text())
            )
            - self.fr_margin_pixel
        )
        self.draw()
        self.setFocus()

    def onFrequencyMarkerUnitMaxEdited(self):
        self.fr_marker_max = int(
            self.convert_ppm_to_pts(
                float(self.frequencyMarkerUnitMaxEdit.text())
            )
            - self.fr_margin_pixel
        )
        self.draw()
        self.setFocus()

    def onFrequencyRangeUnitMinEdited(self):
        self.fr_margin_pixel = int(
            self.convert_ppm_to_pts(
                float(self.frequencyRangeUnitMinEdit.text())
            )
        )
        self.detect()
        self.draw()
        self.setFocus()

    def onFrequencyRangeUnitMaxEdited(self):
        self.fr_margin_pixel = int(
            self.convert_ppm_to_pts(
                float(self.frequencyRangeUnitMaxEdit.text())
            )
            - fr_pixels
        )
        self.detect()
        self.draw()
        self.setFocus()

    def onSpeedMarkerPtsMinEdited(self):
        sp_marker_min_param = float(self.speedMarkerPtsMinEdit.text())
        sp_marker_min_unit = (sp_marker_min_param - sp_min) / (sp_max - sp_min)
        sp_marker_min_pixel = sp_marker_min_unit * sp_pixels
        self.sp_marker_min = round(sp_marker_min_pixel)
        self.draw()
        self.setFocus()

    def onSpeedMarkerPtsMaxEdited(self):
        sp_marker_max_param = float(self.speedMarkerPtsMaxEdit.text())
        sp_marker_max_unit = (sp_marker_max_param - sp_min) / (sp_max - sp_min)
        sp_marker_max_pixel = sp_marker_max_unit * sp_pixels
        self.sp_marker_max = round(sp_marker_max_pixel)
        self.draw()
        self.setFocus()

    def onFrequencyMarkerPtsMinEdited(self):
        self.fr_marker_min = (
            int(self.frequencyMarkerPtsMinEdit.text()) - self.fr_margin_pixel
        )
        self.draw()
        self.setFocus()

    def onFrequencyMarkerPtsMaxEdited(self):
        self.fr_marker_max = (
            int(self.frequencyMarkerPtsMaxEdit.text()) - self.fr_margin_pixel
        )
        self.draw()
        self.setFocus()

    def onFrequencyRangePtsMinEdited(self):
        self.fr_margin_pixel = int(self.frequencyRangePtsMinEdit.text())
        self.detect()
        self.draw()
        self.setFocus()

    def onFrequencyRangePtsMaxEdited(self):
        self.fr_margin_pixel = (
            int(self.frequencyRangePtsMaxEdit.text()) - fr_pixels
        )
        self.detect()
        self.draw()
        self.setFocus()

    def detect(self):
        input0 = self.radon[
            :, self.fr_margin_pixel : self.fr_margin_pixel + fr_pixels
        ]
        input0 = (input0[None] - self.add) / self.scale
        # Predicted logits and shifts from model
        ob_logit1 = self.model(input0)
        # Predicted logits and labels
        ob_logit1 = ob_logit1[0]
        ob_label1 = tf.where(ob_logit1 < 0.0, False, True)
        self.ob_logit1 = ob_logit1.numpy()
        self.ob_prob1 = scipy.special.expit(self.ob_logit1)
        self.ob_label1 = ob_label1.numpy()
        ob_blob1, self.ob_blobs1 = scipy.ndimage.label(
            self.ob_label1.T, np.ones((3, 3))
        )
        ob_blob1 = ob_blob1.T

        ob_hot1 = 1 + np.arange(self.ob_blobs1)[:, None, None] == ob_blob1
        ob_prob1 = np.where(ob_hot1, self.ob_prob1, 0.0)
        ob_arg1 = ob_prob1.reshape(-1, 256 * 256).argmax(1)
        self.sp_pixel1, self.fr_pixel1 = np.unravel_index(ob_arg1, (256, 256))

    def draw(self):
        rd_array0 = self.radon[
            :, self.fr_margin_pixel : self.fr_margin_pixel + fr_pixels
        ].numpy()
        rd_array0 = (rd_array0 - self.add) / self.scale / hf_max

        rd_plot0 = self.colormap(rd_array0)
        rd_plot0 = rd_plot0[:, :, :3]
        # Plot markers in blue
        sp_low_marker_idx = max(0, self.sp_marker_min)
        sp_upp_marker_idx = min(RESOLUTION - 1, self.sp_marker_max - 1)
        fr_low_marker_idx = max(0, self.fr_marker_min)
        fr_upp_marker_idx = min(RESOLUTION - 1, self.fr_marker_max - 1)
        rd_plot0[sp_low_marker_idx, :] = [0.0, 0.0, 1.0]
        rd_plot0[sp_upp_marker_idx, :] = [0.0, 0.0, 1.0]
        rd_plot0[:, fr_low_marker_idx] = [0.0, 0.0, 1.0]
        rd_plot0[:, fr_upp_marker_idx] = [0.0, 0.0, 1.0]

        # Plot detected peaks in red
        rd_plot0[self.ob_label1] = [1.0, 0.0, 0.0]
        rd_plot0[self.sp_pixel1, self.fr_pixel1] = [1.0, 1.0, 1.0]

        rd_plot0 = rd_plot0 * 255.0
        rd_plot0 = rd_plot0.astype(np.uint8)

        rd_image0 = QImage(
            rd_plot0.data, 256, 256, 256 * 3, QImage.Format_RGB888
        )
        rd_pixmap0 = QPixmap.fromImage(rd_image0)
        # rd_pixmap0 = rd_pixmap0.scaled(
        #     PIXMAP_PIXELS, PIXMAP_PIXELS, QtCore.Qt.KeepAspectRatio
        # )
        self.rd_widget.setPixmap(rd_pixmap0)

        fr_max0 = rd_array0[self.sp_marker_min : self.sp_marker_max, :].max(0)
        fr_mean0 = rd_array0[self.sp_marker_min : self.sp_marker_max, :].mean(
            0
        )
        fr_min0 = rd_array0[self.sp_marker_min : self.sp_marker_max, :].min(0)

        fr_grid0 = np.linspace(-1.0 / 16.0, 1.0, 16 + 256, endpoint=False)

        fr_plot0 = np.ones((fr_pixels, 16 + 256, 3)) * [
            0.267004,
            0.004874,
            0.329415,
        ]
        fr_plot0[
            np.logical_and(
                fr_mean0[:, None] <= fr_grid0, fr_grid0 < fr_max0[:, None]
            )
        ] = [1.0, 0.0, 0.0]
        fr_plot0[
            np.logical_and(
                fr_min0[:, None] <= fr_grid0, fr_grid0 < fr_mean0[:, None]
            )
        ] = [1.0, 1.0, 0.0]
        fr_plot0[fr_grid0 < fr_min0[:, None]] = [0.0, 1.0, 0.0]

        fr_plot0[:, 16] = [0.0, 0.0, 1.0]
        fr_plot0[:, 32] = [0.0, 0.0, 1.0]
        fr_plot0[fr_low_marker_idx, :] = [0.0, 0.0, 1.0]
        fr_plot0[fr_upp_marker_idx, :] = [0.0, 0.0, 1.0]

        fr_plot0 = np.flip(fr_plot0, 1).transpose((1, 0, 2))
        fr_plot0 = fr_plot0 * 255.0
        fr_plot0 = fr_plot0.astype(np.uint8).copy()

        fr_image0 = QImage(
            fr_plot0.data, 256, 16 + 256, 256 * 3, QImage.Format_RGB888
        )
        fr_pixmap0 = QPixmap.fromImage(fr_image0)
        # fr_pixmap0 = fr_pixmap0.scaled(
        #     PIXMAP_PIXELS, PIXMAP_PIXELS, QtCore.Qt.KeepAspectRatio
        # )
        self.fr_widget.setPixmap(fr_pixmap0)

        sp_max0 = rd_array0[:, self.fr_marker_min : self.fr_marker_max].max(1)
        sp_mean0 = rd_array0[:, self.fr_marker_min : self.fr_marker_max].mean(
            1
        )
        sp_min0 = rd_array0[:, self.fr_marker_min : self.fr_marker_max].min(1)

        sp_grid0 = np.linspace(-1.0 / 16.0, 1.0, 16 + 256, endpoint=False)

        sp_plot0 = np.ones((sp_pixels, 16 + 256, 3)) * [
            0.267004,
            0.004874,
            0.329415,
        ]
        sp_plot0[
            np.logical_and(
                sp_mean0[:, None] <= sp_grid0, sp_grid0 < sp_max0[:, None]
            )
        ] = [1.0, 0.0, 0.0]
        sp_plot0[
            np.logical_and(
                sp_min0[:, None] <= sp_grid0, sp_grid0 < sp_mean0[:, None]
            )
        ] = [1.0, 1.0, 0.0]
        sp_plot0[sp_grid0 < sp_min0[:, None]] = [0.0, 1.0, 0.0]

        sp_plot0[:, 16] = [0.0, 0.0, 1.0]
        sp_plot0[:, 32] = [0.0, 0.0, 1.0]
        sp_plot0[sp_low_marker_idx, :] = [0.0, 0.0, 1.0]
        sp_plot0[sp_upp_marker_idx, :] = [0.0, 0.0, 1.0]

        sp_plot0 = np.flip(sp_plot0, 1)
        sp_plot0 = sp_plot0 * 255.0
        sp_plot0 = sp_plot0.astype(np.uint8).copy()

        sp_image0 = QImage(
            sp_plot0.data, 16 + 256, 256, (16 + 256) * 3, QImage.Format_RGB888
        )
        sp_pixmap0 = QPixmap.fromImage(sp_image0)
        # sp_pixmap0 = sp_pixmap0.scaled(
        #     PIXMAP_PIXELS, PIXMAP_PIXELS, QtCore.Qt.KeepAspectRatio
        # )
        self.sp_widget.setPixmap(sp_pixmap0)

        sp_mask = np.logical_and(
            self.sp_marker_min <= sp_offset, sp_offset < self.sp_marker_max
        )
        fr_mask = np.logical_and(
            self.fr_marker_min <= fr_offset, fr_offset < self.fr_marker_max
        )

        mask = np.logical_and(sp_mask, fr_mask)
        mask = np.logical_and(mask, self.ob_label1)

        sp_pixel = sp_offset[mask].numpy().mean() + 0.5
        sp_unit = sp_pixel / 256.0
        sp_param = sp_unit * (sp_max - sp_min) + sp_min
        fr_pixel = fr_offset[mask].numpy().mean() + 0.5 + self.fr_margin_pixel

        # sp_unit = (
        #     np.sign(sp_pixel)
        #     * self.ppm_scale[int(round(abs(sp_pixel)))]
        #     * 1000
        #     if not np.isnan(sp_pixel)
        #     else np.nan
        # )
        fr_unit = (
            self.ppm_scale[int(round(fr_pixel))]
            if not np.isnan(fr_pixel)
            else np.nan
        )

        self.speedEditUnit.setText("%.4f" % (sp_param,))
        self.frequencyEditUnit.setText("%.4f" % (fr_unit,))

        self.speedEditPts.setText("%.1f" % (sp_pixel,))
        self.frequencyEditPts.setText("%.1f" % (fr_pixel,))

        self.subtractEdit.setText("%.3f" % (self.add,))
        self.divideEdit.setText("%.3f" % (self.scale,))

        sp_marker_min_unit = self.sp_marker_min / sp_pixels
        sp_marker_min_param = sp_marker_min_unit * (sp_max - sp_min) + sp_min
        sp_marker_max_unit = self.sp_marker_max / sp_pixels
        sp_marker_max_param = sp_marker_max_unit * (sp_max - sp_min) + sp_min

        # Update unit edits.
        self.speedMarkerUnitMinEdit.setText(
            "%.3f"
            % (
                self.convert_pts_to_ppm(
                    sp_marker_min_param + self.convert_ppm_to_pts(0),
                )
                * 1000
            )
        )
        self.speedMarkerUnitMaxEdit.setText(
            "%.3f"
            % (
                self.convert_pts_to_ppm(
                    sp_marker_max_param + self.convert_ppm_to_pts(0),
                )
                * 1000
            )
        )

        self.frequencyMarkerUnitMinEdit.setText(
            "%.3f"
            % self.convert_pts_to_ppm(
                self.fr_margin_pixel + self.fr_marker_min,
            )
        )
        self.frequencyMarkerUnitMaxEdit.setText(
            "%.3f"
            % self.convert_pts_to_ppm(
                self.fr_margin_pixel + self.fr_marker_max,
            )
        )
        self.frequencyRangeUnitMinEdit.setText(
            "%.3f"
            % self.convert_pts_to_ppm(
                self.fr_margin_pixel,
            )
        )
        self.frequencyRangeUnitMaxEdit.setText(
            "%.3f"
            % self.convert_pts_to_ppm(
                self.fr_margin_pixel + fr_pixels,
            )
        )

        # Update pts edits.
        self.speedMarkerPtsMinEdit.setText("%.2f" % (sp_marker_min_param,))
        self.speedMarkerPtsMaxEdit.setText("%.2f" % (sp_marker_max_param,))

        self.frequencyMarkerPtsMinEdit.setText(
            "%i" % (self.fr_margin_pixel + self.fr_marker_min,)
        )
        self.frequencyMarkerPtsMaxEdit.setText(
            "%i" % (self.fr_margin_pixel + self.fr_marker_max,)
        )
        self.frequencyRangePtsMinEdit.setText("%i" % (self.fr_margin_pixel,))
        self.frequencyRangePtsMaxEdit.setText(
            "%i" % (self.fr_margin_pixel + fr_pixels,)
        )

    def keyPressEvent(self, event):
        key = event.key()
        minus = key == Qt.Key_Minus
        plus = key == Qt.Key_Plus or key == Qt.Key_Equal
        left = key == Qt.Key_Left
        right = key == Qt.Key_Right
        down = key == Qt.Key_Down
        up = key == Qt.Key_Up
        modifiers = event.modifiers()
        shift = modifiers & Qt.ShiftModifier
        control = modifiers & Qt.ControlModifier
        if not shift and not control and minus:
            self.scale /= 1.189207115
            self.detect()
        elif not shift and not control and plus:
            self.scale *= 1.189207115
            self.detect()
        elif (shift or control) and minus:
            self.add -= self.scale / 16.0
            self.detect()
        elif (shift or control) and plus:
            self.add += self.scale / 16.0
            self.detect()
        elif not shift and not control and left:
            self.fr_margin_pixel -= 1
            self.detect()
        elif not shift and not control and right:
            self.fr_margin_pixel += 1
            self.detect()
        elif not shift and control and left:
            self.fr_marker_min += 1
            self.fr_marker_max -= 1
        elif not shift and control and right:
            self.fr_marker_min -= 1
            self.fr_marker_max += 1
        elif not shift and control and down:
            self.sp_marker_min += 1
            self.sp_marker_max -= 1
        elif not shift and control and up:
            self.sp_marker_min -= 1
            self.sp_marker_max += 1
        elif shift and not control and left:
            self.fr_marker_min -= 1
            self.fr_marker_max -= 1
        elif shift and not control and right:
            self.fr_marker_min += 1
            self.fr_marker_max += 1
        elif shift and not control and down:
            self.sp_marker_min += 1
            self.sp_marker_max += 1
        elif shift and not control and up:
            self.sp_marker_min -= 1
            self.sp_marker_max -= 1
        elif key == Qt.Key_Space:
            self.sp_marker_min = 0
            self.sp_marker_max = 256
            self.fr_marker_min = 0
            self.fr_marker_max = 256
        else:
            return
        self.draw()


def main():
    fid, ppm_scale = read_fid_and_scale()
    radon = radon_transform(fid)
    model = get_detector_model()

    slice_start = 20775

    picker = Picker(radon, model, ppm_scale, start_slice_idx=slice_start)
    picker.show()

    app.exec()


if __name__ == "__main__":
    main()
