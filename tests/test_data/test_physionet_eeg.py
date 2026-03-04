# A testing file for the PhysioNet Sleep EDF EEG data engineering code

import numpy as np
import pytest

from data.physionet_eeg import (
    select_subjects,
    map_sleep_stage,
    subdivide_epoch,
    compute_log_spectrum,
    apply_nmf
)


def test_select_subjects_default():
    subjects = select_subjects()
    assert isinstance(subjects, list)
    assert len(subjects) == 20  # default all subjects
    assert all(isinstance(s, int) for s in subjects)


def test_select_subjects_n_subjects_and_ids():
    subjects = select_subjects(n_subjects=5, random_state=0)
    assert len(subjects) == 5
    # Override by subject_ids
    subjects2 = select_subjects(subject_ids=[1, 3, 7])
    assert subjects2 == [1, 3, 7]


@pytest.mark.parametrize("desc, expected", [
    ("Sleep stage W", "awake"),
    ("Sleep stage R", "REM"),
    ("Sleep stage 1", "non-REM"),
    ("Sleep stage 2", "non-REM"),
    ("Sleep stage 3", "non-REM"),
    ("Sleep stage 4", "non-REM"),
    ("Unknown stage", None)
])


def test_map_sleep_stage(desc, expected):
    assert map_sleep_stage(desc) == expected


def test_subdivide_epoch_shapes():
    fs = 100
    epoch_data = np.ones((1, 1, 30*fs))
    segments = subdivide_epoch(epoch_data, fs)
    assert len(segments) == 3
    for seg in segments:
        assert seg.shape[0] == 10*fs


def test_compute_log_spectrum_non_negative():
    fs = 100
    signal = np.sin(2*np.pi*1*np.arange(0,10*fs)/fs)
    log_spec = compute_log_spectrum(signal, fs=fs)
    assert isinstance(log_spec, np.ndarray)
    assert log_spec.ndim == 1
    assert np.all(log_spec >= 0)  # log1p of abs STFT is non-negative


def test_apply_nmf_output_shape():
    X = np.abs(np.random.randn(10, 50))
    W = apply_nmf(X, n_components=5)
    assert isinstance(W, np.ndarray)
    assert W.shape == (10, 5)
    assert np.all(W >= 0)  # NMF produces non-negative coefficients
