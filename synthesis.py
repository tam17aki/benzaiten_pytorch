# -*- coding: utf-8 -*-
"""Demonstration script for melody generation using pretrained model.

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
import os

import matplotlib.pyplot as plt
import midi2audio
import numpy as np
import torch
from hydra import compose, initialize
from scipy.special import softmax

from model import Seq2SeqMelodyComposer
from util import (
    calc_durations,
    calc_notenums_from_pianoroll,
    calc_xy,
    chord_seq_to_chroma,
    extract_seq,
    make_chord_seq,
    make_empty_pianoroll,
    make_midi,
    read_chord_file,
)


@torch.no_grad()
def generate_melody(cfg, model, chroma_vec, device):
    """Perform a inference step to generate melody (piano-roll) data.

    Args:
        chroma_vec : sequence of many-hot (chroma) vectors

    Returns:
        piano_roll (numpy.ndarray): generated melody
    """
    piano_roll = make_empty_pianoroll(
        chroma_vec.shape[0],
        cfg.feature.notenum_thru,
        cfg.feature.notenum_from,
    )
    beat_width = cfg.feature.n_beats * cfg.feature.beat_reso
    for i in range(0, cfg.feature.melody_length, cfg.feature.unit_measures):
        onehot_vectors, chord_vectors = extract_seq(
            i, piano_roll, chroma_vec, cfg.feature.unit_measures, beat_width
        )
        feature, label = calc_xy(onehot_vectors, chord_vectors)
        feature = torch.from_numpy(feature).to(device).float()
        feature = feature.unsqueeze(0)
        y_new = model(feature)
        y_new = y_new.to("cpu").detach().numpy().copy()
        y_new = softmax(y_new, axis=-1)
        index_from = i * (cfg.feature.n_beats * cfg.feature.beat_reso)
        piano_roll[index_from : index_from + y_new[0].shape[0], :] = y_new[0]

    plt.matshow(np.transpose(piano_roll))
    png_file = os.path.join(
        cfg.benzaiten.root_dir, cfg.benzaiten.adlib_dir, cfg.demo.pianoroll_file
    )
    plt.savefig(png_file)
    return piano_roll


def generate_midi(cfg, model, chord_file, device):
    """Synthesize melody with a trained model.

    Args:
        chord_file: a file of chord sequence (csv)

    Returns:
        midi: generated midi data
    """
    chord_prog = read_chord_file(
        chord_file, cfg.feature.melody_length, cfg.feature.n_beats
    )
    chord_seq = make_chord_seq(
        chord_prog,
        cfg.feature.n_beats,
        cfg.feature.n_beats,
        cfg.feature.beat_reso,
    )
    chroma_vec = chord_seq_to_chroma(chord_seq)
    piano_roll = generate_melody(cfg, model, chroma_vec, device)
    notenums = calc_notenums_from_pianoroll(piano_roll, cfg.feature.notenum_from)
    notenums, durations = calc_durations(notenums)
    midi = make_midi(cfg, notenums, durations)
    return midi


def main(cfg):
    """Perform ad-lib melody synthesis."""

    # setup network and load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckkpt_dir = os.path.join(cfg.benzaiten.root_dir, cfg.demo.chkpt_dir)
    checkpoint = os.path.join(ckkpt_dir, cfg.demo.chkpt_file)
    model = Seq2SeqMelodyComposer(cfg, device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()  # turn on eval mode

    # generate ad-lib melody in midi format
    chord_file = os.path.join(
        cfg.benzaiten.root_dir, cfg.benzaiten.adlib_dir, cfg.demo.chord_file
    )
    midi_file = os.path.join(
        cfg.benzaiten.root_dir, cfg.benzaiten.adlib_dir, cfg.demo.midi_file
    )
    midi = generate_midi(cfg, model, chord_file, device)
    midi.save(midi_file)

    # port midi to wav
    wav_file = os.path.join(
        cfg.benzaiten.root_dir, cfg.benzaiten.adlib_dir, cfg.demo.wav_file
    )
    fluid_synth = midi2audio.FluidSynth(sound_font=cfg.demo.sound_font)
    fluid_synth.midi_to_audio(midi_file, wav_file)


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")

    main(config)
