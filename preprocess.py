# -*- coding: utf-8 -*-
"""Preprocess script for Benzaiten Starter Kit ver. 1.0

Copyright (C) 2022 by 北原 鉄朗 (Tetsuro Kitahara)
Copyright (C) 2022 by Akira TAMAMORI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import glob
import os
import subprocess

import joblib
import music21
import numpy as np
from hydra import compose, initialize
from progressbar import progressbar as prg

from util import (
    add_rest_nodes,
    chord_seq_to_chroma,
    divide_seq,
    make_note_and_chord_seq_from_musicxml,
    note_seq_to_onehot,
)


def get_music_xml(cfg):
    """Download Omnibook MusicXML data for training."""
    xml_url = cfg.preprocess.xml_url
    xml_dir = os.path.join(cfg.benzaiten.root_dir, cfg.benzaiten.xml_dir)
    os.makedirs(xml_dir, exist_ok=True)

    subprocess.run("echo -n Download Omnibook MusicXML ... ", text=True, shell=True)

    command = "wget " + "-P " + "/tmp/" + " " + xml_url
    subprocess.run(command, text=True, shell=True, capture_output=True)

    zip_file = os.path.basename(xml_url)
    command = "cd " + xml_dir + "; " + "unzip " + "/tmp/" + zip_file
    subprocess.run(command, text=True, shell=True, capture_output=True)

    command = "mv " + xml_dir + "Omnibook\\ xml/*.xml " + xml_dir
    subprocess.run(command, text=True, shell=True)

    command = "rm -rf " + xml_dir + "Omnibook\\ xml"
    subprocess.run(command, text=True, shell=True)

    command = "rm -rf " + xml_dir + "__MACOSX"
    subprocess.run(command, text=True, shell=True)

    print(" done.")


def extract_features(cfg):
    """Extract features."""
    data_all = []
    label_all = []
    xml_dir = os.path.join(cfg.benzaiten.root_dir, cfg.benzaiten.xml_dir)
    os.makedirs(xml_dir, exist_ok=True)
    for xml_file in prg(
        glob.glob(xml_dir + "/*.xml"), prefix="Extract features from MusicXML: "
    ):
        score = music21.converter.parse(xml_file)
        key = score.analyze("key")
        if key.mode == cfg.feature.key_mode:
            inter = music21.interval.Interval(
                key.tonic, music21.pitch.Pitch(cfg.feature.key_root)
            )
            score = score.transpose(inter)
            note_seq, chord_seq = make_note_and_chord_seq_from_musicxml(
                score,
                cfg.feature.total_measures,
                cfg.feature.n_beats,
                cfg.feature.beat_reso,
            )
            onehot_seq = add_rest_nodes(
                note_seq_to_onehot(
                    note_seq,
                    cfg.feature.notenum_thru,
                    cfg.feature.notenum_from,
                )
            )
            chroma_seq = chord_seq_to_chroma(chord_seq)
            divide_seq(cfg, onehot_seq, chroma_seq, data_all, label_all)

    return np.array(data_all), np.array(label_all)


def save_features(cfg, data_all, label_all):
    """Save feature vectors."""
    feats_dir = os.path.join(cfg.benzaiten.root_dir, cfg.benzaiten.feat_dir)
    os.makedirs(feats_dir, exist_ok=True)
    feat_file = os.path.join(feats_dir, cfg.preprocess.feat_file)
    joblib.dump({"data": data_all, "label": label_all}, feat_file)
    print("Save extracted features to " + feat_file)


def get_backing_chord(cfg):
    """Download backing file (midi) and chord file (csv)."""
    g_drive_url = '"https://drive.google.com/uc?export=download&id="'
    adlib_dir = os.path.join(cfg.benzaiten.root_dir, cfg.benzaiten.adlib_dir)
    os.makedirs(adlib_dir, exist_ok=True)

    backing_url = g_drive_url + cfg.demo.backing_fid
    backing_file = os.path.join(adlib_dir, cfg.demo.backing_file)
    chord_url = g_drive_url + cfg.demo.chord_fid
    chord_file = os.path.join(adlib_dir, cfg.demo.chord_file)

    subprocess.run("echo -n Download backing file for demo ... ", text=True, shell=True)
    command = "wget " + backing_url + " -O " + backing_file
    subprocess.run(command, text=True, shell=True, capture_output=True)
    print(" done.")

    subprocess.run("echo -n Download chord file for demo ... ", text=True, shell=True)
    command = "wget " + chord_url + " -O " + chord_file
    subprocess.run(command, text=True, shell=True, capture_output=True)
    print(" done.")


def main(cfg):
    """Perform preprocess."""
    # Download Omnibook MusicXML
    get_music_xml(cfg)

    # Extract features from MusicXML
    data_all, label_all = extract_features(cfg)

    # Save extracted features.
    save_features(cfg, data_all, label_all)

    # Download backing file (midi) and chord file (csv)
    get_backing_chord(cfg)


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")

    main(config)
