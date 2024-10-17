import argparse
import gc
import json
import os
from copy import deepcopy

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import yaml
from scipy.signal import ShortTimeFFT
from tqdm import tqdm

PARSED_DATA_DIR = './data_parsed'
PARSED_METADATA_FILENAME = 'parsed_metadata.json'
OUTPUT_DIR = './data_preprocessed'
PREPROCESSED_METADATA_FILENAME = 'preprocessed_metadata.json'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_arguments():

    parser = argparse.ArgumentParser(
        description='data preprocessing config path')

    parser.add_argument(
        '--config-path',
        '-c',
        action='store',
        dest='config_path',
        help='config path',
        default="./data_preprocessing_config.yml",
        required=False)

    args = parser.parse_args()

    return args


# scipy's window functions
# window_fn_names: list of names of available scipy window functions
window_fn_names = scipy.signal.windows.__all__


def get_window_fn(name: str = 'hann', **kwargs):
    '''Gets scipy window function with given window function name
    and arguments. The window function is used in computing short-time
    Fourier transforms (STFTs).

    Args:
        name (str, optional): one of window_fn_names. Defaults to 'hann'.
        **kwargs: passed to the window function

    Returns:
        window_fn: window function
    '''
    window_fn_map = {key: scipy.signal.windows.__dict__[key]
                     for key in window_fn_names}
    window_fn = window_fn_map[name](**kwargs)
    return window_fn


def compute_fft(x: np.ndarray, fs: float,
                return_spectrum: bool = True):
    '''Compute discrete (fast) Fourier transform (FFT) of a possibly
    multivariate signal, channel-wise.
    Option to return the FFT or
    spectrum (absolute-square of FFT coefficients).

    Args:
        x (np.ndarray): a univariate/multivariate signal.
            If multivariate, x.shape == (signal_length, num_channels)
        fs (float): sampling frequency in Hz
        return_spectrum (bool, optional): return spectrum,
            otherwise return fft.
            Defaults to True.

    Returns:
        fft_freq (np.ndarray): Fourier frequencies of the FFT
        spectrum (np.ndarray) if return_spectrum,
        else fft (np.ndarray)
    '''
    N = x.shape[0]  # signal_length
    fft = scipy.fft.fft(x, axis=0)
    fft_freq = np.linspace(0, fs * (N - 1) / N, N)
    if return_spectrum:
        spectrum = np.square(np.abs(fft))
        return fft_freq, spectrum
    else:
        return fft_freq, fft


def convert_window_hop(window_length_sec: float,
                       hop_sec: float,
                       fs: float):
    '''Converts window (frame) length and hop in seconds to
    in the number of data points/timesteps

    Args:
        window_length_sec (float): window length in second
        hop_sec (float): hop length in second
        fs (float): sampling frequency in Hz

    Returns:
        mfft (int): = n_fft = frame size
            = number of data points in each frame (window)
        hop (int): hop = stride in the number of timesteps
    '''
    mfft = int(window_length_sec * fs)
    hop = int(hop_sec * fs)
    return mfft, hop


def compute_stft(x: np.ndarray,
                 fs: float,
                 mfft: int, hop: int,
                 window_fn_name: str = 'hann',
                 window_fn_args: dict = {},
                 return_spectrogram: bool = True,
                 start_time: float = 0.,
                 ):
    '''Computes short-time Fourier transform (STFT) of a given possible
    multivariate signal, using scipy.
    Option to return the STFT or
    spectrogram (absolute-square of STFT coefficients).

    Args:
        x (np.ndarray): a univariate/multivariate signal.
            If multivariate, x.shape == (signal_length, num_channels)
        fs (float): sampling frequency in Hz
        mfft (int): = n_fft = frame size
            = number of data points in each frame (window)
        hop (int): hop = stride in the number of timesteps
        window_fn_name (str): name of a scipy window function.
            Defaults to 'hann'.
        window_fn_args (dict): arguments passed to the window function.
            Defaults to {}
        return_spectrogram (bool, optional):
            return spectrogram, otherwise return stft.
            Defaults to True.
        start_time (float, optional): start time of the signal.
            Defaults to 0.

    Returns:
        stft_freq (np.ndarray): Fourier frequencies of the STFT
        stft_time (np.ndarray): time steps of the STFT
        spectrogram (np.ndarray) if return_spectrogram,
        else stft (np.ndarray)
        SFT: scipy.signal.ShortTimeFFT object, for possible later use
            e.g. computing cepstrum
    '''
    x_ = x
    if x.ndim == 1:
        x_ = np.expand_dims(x, axis=-1)
    # window function
    window_fn = get_window_fn(name=window_fn_name, **window_fn_args)
    # create scipy STFT object
    SFT = scipy.signal.ShortTimeFFT(win=window_fn, fs=fs,
                                    hop=hop, mfft=mfft)
    # compute stft of the given signal
    # squeeze channel in case of univariate signal
    stft = SFT.stft(x_, axis=0)
    if x.ndim == 1:
        stft = stft.squeeze()
    # The Fourier frequencies are given by a rescaling as follows
    # stft_freq = np.arange(0, int(1 + n_fft / 2)) * fs / n_fft
    stft_freq = SFT.f
    # time steps
    stft_time = start_time + np.arange(stft.shape[-1]) * SFT.delta_t
    if return_spectrogram:
        spectrogram = np.square(np.abs(stft))
        return stft_freq, stft_time, spectrogram, SFT
    else:
        return stft_freq, stft_time, stft, SFT


def compute_stft_librosa(x: np.ndarray,
                         fs: float,
                         mfft: int, hop: int,
                         window_fn_name: str = 'hann',
                         window_fn_args: dict = {},
                         return_spectrogram: bool = True,
                         start_time: float = 0.,
                         ):
    '''Computes short-time Fourier transform (STFT) of a given **univariate**
    signal, using librosa.

    Args:
        x (np.ndarray): a univariate signal
        fs (float): sampling frequency in Hz
        mfft (int): = n_fft = frame size
            = number of data points in each frame (window)
        hop (int): hop = stride in the number of timesteps
        window_fn_name (str): name of a scipy window function.
            Defaults to 'hann'.
        window_fn_args (dict): arguments passed to the window function.
            Defaults to {}
        return_spectrogram (bool, optional):
            return spectrogram, otherwise return stft.
            Defaults to True.
        start_time (float, optional): start time of the signal.
            Defaults to 0.

    Returns:
        stft_freq (np.ndarray): Fourier frequencies of the STFT
        stft_time (np.ndarray): time steps of the STFT
        spectrogram (np.ndarray) if return_spectrogram,
        else stft (np.ndarray)
        None: placeholder to align with outputs of compute_stft
    '''
    # window function
    window_fn = get_window_fn(name=window_fn_name, **window_fn_args)
    n_fft = mfft
    # compute stft of the given signal
    stft = librosa.stft(x, n_fft=n_fft, hop_length=hop,
                        window=window_fn)
    # The Fourier frequencies are given by a rescaling as follows
    stft_freq = np.arange(0, int(1 + n_fft / 2)) * fs / n_fft
    # time steps
    stft_time = start_time + np.arange(stft.shape[1]) / fs
    if return_spectrogram:
        spectrogram = np.square(np.abs(stft))
        return stft_freq, stft_time, spectrogram, None
    else:
        return stft_freq, stft_time, stft, None


def plot_spectrogram(input_data: np.ndarray,
                     stft_freq: np.ndarray,
                     stft_time: np.ndarray,
                     input_is_spectrogram: bool = True,
                     plot_channel: int = 0,
                     ylim: tuple = None
                     ):
    '''Plots spectrogram

    Args:
        input_data (np.ndarray): either spectrogram or stft.
            User needs to set input_is_spectrogram accordingly
        stft_freq (np.ndarray): Fourier frequencies of the STFT
        stft_time (np.ndarray): time steps of the STFT
        input_is_spectrogram (bool, optional): Defaults to True.
        ylim (tuple, optional): Defaults to None.
    '''
    input_data_ = input_data
    if not input_is_spectrogram:
        input_data_ = np.square(np.abs(input_data_))
    if input_data_.ndim == 2:
        input_data_ = np.expand_dims(input_data_, axis=1)
    plt.pcolormesh(stft_time, stft_freq, input_data_[:, plot_channel, :])
    if ylim:
        plt.ylim(ylim)
    plt.title('Power spectrum')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.colorbar()
    plt.show()


def create_sequences(x: np.ndarray, sequence_length: int = 100,
                     stride: int = None, padding: bool = True,
                     pad_mode: str = 'median'):
    """Construct sliding window view (fixed-length sequences with a given
    stride) on the first axis of a numpy array,
    pads or drops the 'loose end'.

    Args:
        x (numpy.ndarray): numpy.ndarray of shape (n,...). Sliding window view
            is constructed on the first axis
        sequence_length (int): Defaults to 100
        stride (int): if None, stride = sequence_length (no overlap between
            sequences). Defaults to None
        padding (bool): whether to pad the ending sequence. If False, the
            'loose end' will be truncated. Defaults to True
        pad_mode (str): np.pad mode. Defaults to 'median'
    returns:
        sequences (numpy.ndarray):
            None if no sequences,
            shape==(<num_sequences>, <sequence_length>, ...) if x.ndim > 1,
            shape==(<num_sequences>, <sequence_length>, 1) if x.ndim == 1
    """
    # empty numpy array of any shape
    if min(x.shape) == 0:
        return None
    # unsqueeze x along the last dimension
    x_ = x if x.ndim > 1 else np.expand_dims(x, axis=-1)
    sequences = []
    # set stride to sequence_length if not specified,
    # i.e. no overlap between sequences
    if stride is None:
        stride = sequence_length

    # number of non-out-of-bound sequences
    n_full_seq = (x_.shape[0] - sequence_length) // stride + 1
    loose_tail = ((x_.shape[0] - sequence_length) % stride > 0.)
    # we are windowing on the first dimension of x
    # partition into sequences
    # construct sequences from x
    for i in range(n_full_seq):
        sequence_ = x_.take(
            indices=range(i * stride, (i * stride + sequence_length)), axis=0)
        sequences.append(sequence_)
    # the last sequence may be shorter than sequence_length
    # (loose_tail == True)
    # if padding, pad it and append to sequences
    # otherwise skip (truncate)
    if loose_tail and padding:
        sequence_ = x_.take(indices=range(n_full_seq * stride, x_.shape[0]),
                            axis=0)
        # only pad along the first dimension (the one we are partitioning)
        npad = [(0, sequence_length - sequence_.shape[0]) if n == 0 else (0, 0)
                for n in range(sequence_.ndim)]
        sequence_ = np.pad(sequence_, npad, mode=pad_mode)
        sequences.append(sequence_)
    if len(sequences) == 0:
        return None
    sequences = np.stack(sequences)
    return sequences


def build_sequence_spectrum(x: np.ndarray,
                            sequence_length: int, sequence_stride: int,
                            fs: float
                            ):
    '''Segments a possibly multivariate signal into segments (sequences)
    with a given stride, followed by computing the fast Fourier transform (FFT)
    spectrum of each sequence, and computing the spectrum
    (absolute-square of FFT coefficients).

    Args:
        x (np.ndarray): a univariate/multivariate signal.
            If multivariate, x.shape == (signal_length, num_channels)
        sequence_length (int): length of sequence (in number of data points)
        sequence_stride (int): sequence stride (in number of data points)
        fs (float): sampling frequency in Hz

    Returns:
        fft_freq (np.ndarray): Fourier frequencies of the FFT
        seq_spectrum (list of np.ndarray): list of (channel last)
        spectrum, one for each sequence

    '''
    sequences = create_sequences(x,
                                 sequence_length=sequence_length,
                                 stride=sequence_stride)
    seq_spectrum = []
    fft_freq = None
    for n in range(sequences.shape[0]):
        gc.collect()
        x_ = sequences[n, ...]
        fft_freq_, spectrum_ = compute_fft(x_, fs=fs, return_spectrum=True)
        seq_spectrum.append(spectrum_)
    fft_freq = fft_freq_
    return fft_freq, seq_spectrum


def build_sequence_spectrogram(x: np.ndarray,
                               sequence_length: int, sequence_stride: int,
                               fs: float, mfft: int, hop: int,
                               window_fn_name: str = 'hann',
                               window_fn_args: dict = {},
                               channel_first: bool = True
                               ):
    '''Segments a possibly multivariate signal into segments (sequences)
    with a given stride, followed by computing the short-time Fourier transform
    (STFT) spectrogram (absolute-square of STFT coefficients) of each sequence.

    Args:
        x (np.ndarray): a univariate/multivariate signal.
            If multivariate, x.shape == (signal_length, num_channels)
        sequence_length (int): length of sequence (in number of data points)
        sequence_stride (int): sequence stride (in number of data points)
        fs (float): sampling frequency in Hz
        mfft (int): = n_fft = frame size
            = number of data points in each frame (window)
        hop (int): hop = stride in the number of timesteps
        window_fn_name (str): name of a scipy window function.
            Defaults to 'hann'.
        window_fn_args (dict): arguments passed to the window function.
            Defaults to {}
        channel_first (bool): convert resulting spectrum to channel-first.
            Defaults to True

    Returns:
        stft_freq (np.ndarray): Fourier frequencies of the STFT
        stft_time (np.ndarray): time steps of the STFT
        seq_spectrogram (np.ndarray): list of (channel_first) spectrogram,
            one for each sequence
    '''
    sequences = create_sequences(x,
                                 sequence_length=sequence_length,
                                 stride=sequence_stride)
    seq_spectrogram = []
    stft_freq = None
    stft_time = None
    for n in range(sequences.shape[0]):
        x_ = sequences[n, ...]
        stft_freq_, stft_time_, spectrogram, _ = \
            compute_stft(x_,
                         fs=fs, mfft=mfft, hop=hop,
                         window_fn_name=window_fn_name,
                         window_fn_args=window_fn_args,
                         return_spectrogram=True)
        if x_.ndim > 1 and channel_first:
            spectrogram = np.swapaxes(spectrogram, 0, 1)
        seq_spectrogram.append(spectrogram)
    stft_freq = stft_freq_
    stft_time = stft_time_
    return stft_freq, stft_time, seq_spectrogram


def preprocess_data(parsed_data_dir: str = PARSED_DATA_DIR,
                    parsed_metadata_filename: str = PARSED_METADATA_FILENAME,
                    output_dir: str = OUTPUT_DIR,
                    preprocessed_metadata_filename:
                        str = 'preprocessed_metadata.json',
                    index_column: str = 'index',
                    data_columns: list = ["DE_time", "FE_time"],
                    preprocessing_specs: dict = {
                        'sequence_length': 11500,
                        'sequence_stride': 345,
                        'hop': 54,
                        'mfft': 452,
                        'window_fn_name': 'hann',
                        'window_fn_args': {'M': 104},
                        'channel_first': True
                    }):
    '''Loads parsed data. For each bearing cycle,
    segments a possibly multivariate signal into segments (sequences)
    with a given stride, followed by computing the short-time Fourier transform
    (STFT) spectrogram of each sequence.
    TODO: add spectrum (FFT; 1d not 2d) support in addition to spectrogram (2d)
    The preprocessing_specs follow:
    https://www.sciencedirect.com/science/article/pii/S0888327021010499
    https://arxiv.org/abs/2407.14625
    except for the choice of the window function (we use hann).

    Args:
        parsed_data_dir (str, optional): from the data_parse step.
            Defaults to PARSED_DATA_DIR.
        parsed_metadata_filename (str, optional): from the data_parse step.
            Defaults to PARSED_METADATA_FILENAME.
        output_dir (str, optional): Defaults to OUTPUT_DIR.
        preprocessed_metadata_filename (str, optional):
            Defaults to 'preprocessed_metadata.json'.
        index_column (str, optional): timestep column in the parsed dataframe.
            Defaults to 'index'.
        data_columns (list, optional): time series columns for which to compute
            spectrogram/spectrum in parsed data.
            Defaults to ["DE_time", "FE_time"].
        preprocessing_specs (_type_, optional):
            Defaults to {
            'sequence_length': 11500, 'sequence_stride': 345,
            'hop': 54, 'mfft': 452,
            'window_fn_name': 'hann', 'window_fn_args': {'M': 104},
            'channel_first': True}.

    Returns:
        preprocessed_metadata (dict): same as parsed_metadata
            with new field
            'preprocessed_data_path': os.path.join(output_dir,
                                                   f'{cycle_id}.npz')
    '''
    os.makedirs(output_dir, exist_ok=True)

    # load parsed metadata
    with open(os.path.join(parsed_data_dir,
                           parsed_metadata_filename), 'r') as f:
        metadata = json.load(f)

    # build spectrograms for each cycle
    # save to npz file
    preprocessed_metadata = []
    for cycle_ in tqdm(metadata):
        metadata_preprocessed_ = deepcopy(cycle_)
        cycle_id = cycle_['cycle_id']
        # load cycle's bearing time series from parsed csv
        # set index_column as index and keep only the data_column
        x_ = (pd.read_csv(cycle_['parsed_data_path'])
              .set_index(index_column).sort_index()[data_columns]
              .to_numpy())
        preprocessing_specs_ = deepcopy(preprocessing_specs)
        # get sampling frequency fs from metadata
        # and update preprocessing_specs_
        fs = cycle_['sampling_freq_khz'] * 1000
        preprocessing_specs_.update({'fs': fs})
        # compute spectrograms
        stft_freq, stft_time, seq_spectrogram = build_sequence_spectrogram(
            x_, **preprocessing_specs_
        )

        # save Fourier frequencies, timesteps, spectrograms to npz file
        preprocessed_data_path_ = os.path.join(output_dir, f'{cycle_id}.npz')
        metadata_preprocessed_['preprocessed_data_path'] = \
            preprocessed_data_path_
        np.savez(preprocessed_data_path_,
                 stft_freq=stft_freq, stft_time=stft_time,
                 seq_spectrogram=seq_spectrogram)
        # to load: npzfiles = np.load('<filename>.npz')
        # stft_freq = npzfiles['stft_freq']
        # stft_time = npzfiles['stft_time']
        # seq_spectrogram = npzfiles['seq_spectrogram']
        preprocessed_metadata.append(metadata_preprocessed_)

    # save preprocessed metadata
    with open(os.path.join(output_dir, preprocessed_metadata_filename),
              'w') as f:
        json.dump(preprocessed_metadata, f, indent=4)
    return preprocessed_metadata


if __name__ == "__main__":
    # load arguments
    args = get_arguments()
    config_path = args.config_path
    with open(config_path, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)

    # save a copy of the config
    os.makedirs(config['output_dir'], exist_ok=True)
    with open(os.path.join(config['output_dir'], 'config.yml'), 'w') as f:
        yaml.dump(config, f, yaml.SafeDumper)
    _ = config.pop('name')
    preprocessed_metadata = preprocess_data(**config)
