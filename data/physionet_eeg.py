# Code to load and engineer the PhysioNet Sleep EDF EEG data for testing

import mne
import mne.data
import numpy as np
from scipy.signal import stft
from sklearn.decomposition import NMF


def select_subjects(
        n_subjects:int=None,
        subject_ids:list[int]=None,
        random_state:int=42
    ) -> list[int]:
    """
        Get a list of subject ids to sample EEG data for
        Params:
            n_subjects (optional): Number of subjects to sample. Will sample all
                20 if no value and no ids provided
            subject_ids (optional): List of subject ids to sample - will
                override n_subjects if provided
            random_state (optional): A seed if random sampling
        Returns:
            used_subjects: A list of the subject ids to be sampled
    """
    # Choose subject ids to sample
    all_subjects = list(range(20)) # 20 subjects total in the dataset

    rng = np.random.default_rng(random_state)

    if subject_ids is not None:
        used_subjects = subject_ids
    elif n_subjects is not None:
        used_subjects = rng.choice(
            all_subjects,
            size=n_subjects,
            replace=False
        ).tolist()
    else:
        used_subjects = all_subjects
    
    return used_subjects


def load_labelled_epochs(
        subject_id:int,
        channels:list[str]=["EEG Fpz-Cz"] # LAND paper used frontal electrode
    ) -> mne.Epochs:
    """
        Get discretized raw EEG data for given sample
        Params:
            subject_id: The subject we want data for
            channels (optional): The nodes we want data from
                Either EEG Fpz-Cz or EEG Pz-Oz
        Returns:
            The raw annotated EEG data for subject_id divided into 30s epochs

    """
    # Loading raw and annotation data
    paths = mne.datasets.sleep_physionet.age.fetch_data(
        subjects=[subject_id]
    )

    raw = mne.io.read_raw_edf(paths[0][0], preload=True)
    annot = mne.read_annotations(paths[0][1])

    # Annotating raw data with sleep cycle stage string
    raw.set_annotations(annot)

    # Choosing which node to get data from
    raw.pick_channels(channels)
    
    # Filtering for only sleep oscillations
    raw.filter(0.5, 30) 

    # Split the EEG data into 30s intervals - hardcoded to match LAND paper
    events = mne.make_fixed_length_events(raw, duration=30)
    epochs = mne.Epochs(
        raw,
        events,
        tmin=0,
        tmax=30,
        baseline=None,
        preload=True
    )

    return epochs


def map_sleep_stage(description: str) -> str:
    """
        Mapping the annotations to simple labels
        Params:
            description: The annotation to be used to create the label
        Returns:
            label: The simple sleep stage label. One of {REM, awake, non-REM}
    """
    if "Sleep stage W" in description:
        return "awake"
    elif "Sleep stage R" in description:
        return "REM"
    elif any(s in description for s in [
        "Sleep stage 1",
        "Sleep stage 2",
        "Sleep stage 3",
        "Sleep stage 4"
    ]):
        return "non-REM"
    else:
        return None


def subdivide_epoch(epoch_data:np.ndarray, fs=100) -> list[np.ndarray]:
    """
        Break epoch data into subsets based on a given frequency
        Params:
            epoch_data: 3D array with (1, 1, n_times)
                We expect 1 epoch and 1 channel, n_times is the duration * fs
            fs: The sampling frequency for the epoch
        Returns:
            segments: 3 10s subsegments of epoch_data
    """
    signal = epoch_data.squeeze()
    segment_len = 10 * fs # Hardcoding 10s subsegments to match LAND paper
    return [
        signal[i * segment_len: (i+1) * segment_len]
        for i in range(3) # This assumes the epoch has duration=30s
    ]


def compute_log_spectrum(
        signal:np.ndarray,
        fs:int=100,
        nperseg:int=256
    ) -> np.ndarray:
    """
        Compute the Short-Time Fourier Transform on a 10s signal data interval
        and the log-magnitude of the spectrum per-window with 50% window overlap
        Params:
            signal: The subdivided 10s sleep data to be tranformed
            fs: The sampling frequency of the signal data
            nperseg: The number of samples per short-time fourier transform
                calculation
        Returns:
            log_spectrum_mag (np.ndarray): 1D log-amplitude feature vector
    """
    f, t, Zxx = stft(
        signal,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 5 # Hardcoding 50% overlap to match LAND paper
    )

    return np.log1p(np.abs(Zxx)).flatten()


def extract_subject_features(
        subject_id:int
    ) -> tuple[list[np.ndarray], list[str], list[int]]:
    """
        Loads and transforms raw subject data to prepare for feature engineering
        Params:
            subject_id: The subject whose data will be prepped
        Returns:
            feature_list: 1D feature vector for a 10s sleep segment
            labels: Entries are one of {REM, awake, non-REM}
            subject_list: Entries are the input subject_id repeated
    """
    # Load data
    epochs = load_labelled_epochs(subject_id)

    feature_list = []
    labels = []
    subject_list = []

    fs = int(epochs.info['sfreq'])

    # Obtain labels and features for epoch data
    for i, epoch in enumerate(epochs.get_data()):
        label_description = epochs.annotations.description[i]
        label = map_sleep_stage(label_description)

        if label is None:
            continue

        segments = subdivide_epoch(epoch, fs)
        for seg in segments:
            features = compute_log_spectrum(seg, fs)
            feature_list.append(features)
            labels.append(label)
            subject_list.append(subject_id)
        
        return feature_list, labels, subject_id


def apply_nmf(
        X:np.ndarray,
        n_components:int=5 # LAND paper used 5
    ) -> np.ndarray:
    """
        Apply Non-Negative Matrix Factorization of feature data
        Params:
            X: The prepped feature data
            n_components (optional): The desired number of matrix factors
        Returns:
            coefficients: 2D array of component coefficients per sample
    """
    nmf = NMF(
        n_components=n_components,
        init='random',
        n_init=10, # LAND paper uses 10 random starts
        random_state=0,
        max_iter=500
    )
    return nmf.fit_transform(X)


def build_land_sleep_features(
        n_subjects:int=10, # LAND paper used data from 10 subjects
        random_state:int=None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Orchestrate the feature loading and engineering for LAND testing
        Params:
            n_subjects: The number of subjects to sample
            random_state: A seed for reproducibility
        Returns:
            feature_data: The extracted and transformed data ready for LAND use
            labels: The sleep stage labels for the corresponding feature data
                Entries are one of {REM, awake, non-REM}
            subject_id_list: The subject_id for the corresponding feature data
    """
    subjects = select_subjects(n_subjects, random_state)

    features = []
    labels = []
    subject_id_list = []

    for subj in subjects:
        feature_ls, label_ls, subj_ls = extract_subject_features(subj)

        features.extend(feature_ls)
        labels.extend(label_ls)
        subject_id_list.extend(subj_ls)

    feature_data = apply_nmf(np.array(features))

    return (
        feature_data,
        np.array(labels),
        np.array(subject_id_list)
    )
