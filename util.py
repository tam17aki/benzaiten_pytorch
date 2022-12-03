# -*- coding: utf-8 -*-
"""Sample script for Benzaiten Starter Kit ver. 1.0

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
import csv
import os

import mido
import music21
import numpy as np
from omegaconf import DictConfig


def make_note_and_chord_seq_from_musicxml(score, total_measures, n_beats, beat_reso):
    """MusicXMLデータからNote列とChordSymbol列を生成."""
    note_seq = [None] * (total_measures * n_beats * beat_reso)
    chord_seq = [None] * (total_measures * n_beats * beat_reso)
    for element in score.parts[0].elements:
        if isinstance(element, music21.stream.Measure):
            measure_offset = element.offset
            for note in element.notes:
                if isinstance(note, music21.note.Note):
                    onset = measure_offset + note._activeSiteStoredOffset
                    offset = onset + note._duration.quarterLength
                    for i in range(int(onset * beat_reso), int(offset * beat_reso + 1)):
                        note_seq[i] = note

                if isinstance(note, music21.harmony.ChordSymbol):
                    chord_offset = measure_offset + note.offset
                    for i in range(
                        int(chord_offset * beat_reso),
                        int((measure_offset + n_beats) * beat_reso + 1),
                    ):
                        chord_seq[i] = note

    return note_seq, chord_seq


def note_seq_to_onehot(note_seq, notenum_thru, notenum_from):
    """Note列をone-hot vector列（休符はすべて0）に変換."""
    n_note_width = notenum_thru - notenum_from
    n_note_seq = len(note_seq)
    matrix = np.zeros((n_note_seq, n_note_width))
    for i in range(n_note_seq):
        if note_seq[i] is not None:
            matrix[i, note_seq[i].pitch.midi - notenum_from] = 1

    return matrix


def add_rest_nodes(onehot_seq):
    """音符列を表すone-hot vector列に休符要素を追加."""
    rest = 1 - np.sum(onehot_seq, axis=1)
    rest = np.expand_dims(rest, 1)
    return np.concatenate([onehot_seq, rest], axis=1)


def extract_seq(index, onehot_seq, chroma_seq, unit_measures, width):
    """メロディを表すone-hotベクトル、コードを表すmany-hotベクトルの系列に対して、
    UNIT_MEASURES小節分だけ切り出したものを返す."""
    onehot_vectors = onehot_seq[index * width : (index + unit_measures) * width, :]
    chord_vectors = chroma_seq[index * width : (index + unit_measures) * width, :]
    return onehot_vectors, chord_vectors


def chord_seq_to_chroma(chord_seq):
    """ChordSymbol列をmany-hot (chroma) vector列に変換."""
    matrix = np.zeros((len(chord_seq), 12))
    for i, chord in enumerate(chord_seq):
        if chord is not None:
            for note in chord._notes:
                matrix[i, note.pitch.midi % 12] = 1
    return matrix


def read_chord_file(csv_file, melody_length, n_beats):
    """指定された仕様のcsvファイルを読み込んでChordSymbol列を返す"""
    chord_seq = [None] * (melody_length * n_beats)
    with open(csv_file, encoding="utf-8") as file_handler:
        reader = csv.reader(file_handler)
        for row in reader:
            measure_id = int(row[0])  # 小節番号（0始まり）
            if measure_id < melody_length:
                beat_id = int(row[1])  # 拍番号（0始まり、今回は0または2）
                chord_seq[measure_id * 4 + beat_id] = music21.harmony.ChordSymbol(
                    root=row[2], kind=row[3], bass=row[4]
                )
    for i, _chord in enumerate(chord_seq):
        if _chord is not None:
            chord = _chord
        else:
            chord_seq[i] = chord
    return chord_seq


def make_chord_seq(chord_prog, division, n_beats, beat_reso):
    """コード進行からChordSymbol列を生成.

    divisionは1小節に何個コードを入れるか
    """
    time_length = int(n_beats * beat_reso / division)
    seq = [None] * (time_length * len(chord_prog))
    for i, chord in enumerate(chord_prog):
        for _t in range(time_length):
            if isinstance(chord, music21.harmony.ChordSymbol):
                seq[i * time_length + _t] = chord
            else:
                seq[i * time_length + _t] = music21.harmony.ChordSymbol(chord)
    return seq


def make_empty_pianoroll(length, notenum_thru, notenum_from):
    """空（全要素がゼロ）のピアノロールを生成."""
    return np.zeros((length, notenum_thru - notenum_from + 1))


def calc_notenums_from_pianoroll(pianoroll, notenum_from):
    """ピアノロール（one-hot vector列）をノートナンバー列に変換."""
    note_nums = []
    for i in range(pianoroll.shape[0]):
        num = np.argmax(pianoroll[i, :])
        note_num = -1 if num == pianoroll.shape[1] - 1 else num + notenum_from
        note_nums.append(note_num)
    return note_nums


def calc_durations(notenums):
    """連続するノートナンバーを統合して (notenums, durations) に変換."""
    note_length = len(notenums)
    duration = [1] * note_length
    for i in range(note_length):
        k = 1
        while i + k < note_length:
            if notenums[i] > 0 and notenums[i] == notenums[i + k]:
                notenums[i + k] = 0
                duration[i] += 1
            else:
                break
            k += 1
    return notenums, duration


def make_midi(cfg: DictConfig, notenums, durations):
    """MIDIファイルを生成."""
    beat_reso = cfg.feature.beat_reso
    n_beats = cfg.feature.n_beats
    transpose = cfg.feature.transpose
    intro_blank_measures = cfg.feature.intro_blank_measures

    backing_file = os.path.join(
        cfg.benzaiten.root_dir, cfg.benzaiten.adlib_dir, cfg.demo.backing_file
    )
    midi = mido.MidiFile(backing_file)
    track = mido.MidiTrack()
    midi.tracks.append(track)

    var = {
        "init_tick": intro_blank_measures * n_beats * midi.ticks_per_beat,
        "cur_tick": 0,
        "prev_tick": 0,
    }
    for i, notenum in enumerate(notenums):
        if notenum > 0:
            var["cur_tick"] = (
                int(i * midi.ticks_per_beat / beat_reso) + var["init_tick"]
            )
            track.append(
                mido.Message(
                    "note_on",
                    note=notenum + transpose,
                    velocity=100,
                    time=var["cur_tick"] - var["prev_tick"],
                )
            )
            var["prev_tick"] = var["cur_tick"]
            var["cur_tick"] = (
                int((i + durations[i]) * midi.ticks_per_beat / beat_reso)
                + var["init_tick"]
            )
            track.append(
                mido.Message(
                    "note_off",
                    note=notenum + transpose,
                    velocity=100,
                    time=var["cur_tick"] - var["prev_tick"],
                )
            )
            var["prev_tick"] = var["cur_tick"]

    return midi


def calc_xy(onehot_vectors, chord_vectors):
    """メロディを表すone-hotベクトル、コードを表すmany-hotベクトルの系列から、
    モデルの入力、出力用のデータに整えて返す."""
    data = np.concatenate([onehot_vectors, chord_vectors], axis=1)
    label = np.argmax(onehot_vectors, axis=1)
    return data, label


def divide_seq(cfg: DictConfig, onehot_seq, chroma_seq, data_all, label_all):
    """メロディを表すone-hotベクトル、コードを表すmany-hotベクトルの系列から
    モデルの入力、出力用のデータを作成して、配列に逐次格納する."""
    total_measures = cfg.feature.total_measures
    unit_measures = cfg.feature.unit_measures
    beat_width = cfg.feature.n_beats * cfg.feature.beat_reso
    for i in range(0, total_measures, unit_measures):
        onehot_vector, chord_vector = extract_seq(
            i, onehot_seq, chroma_seq, unit_measures, beat_width
        )
        if np.any(onehot_vector[:, 0:-1] != 0):
            data, label = calc_xy(onehot_vector, chord_vector)
            data_all.append(data)
            label_all.append(label)
