import pathlib
import warnings

import audioread
import numpy as np
import resampy
import scipy

import util
import filters

# -- CORE ROUTINES --#
# Load should never be cached, since we cannot verify that the contents of
# 'path' are unchanged across calls.
def load(
    path,
    sr=22050,
    mono=True,
    offset=0.0,
    duration=None,
    dtype=np.float32,
    res_type="kaiser_best",
):
    """Load an audio file as a floating point time series.
    Audio will be automatically resampled to the given rate
    (default ``sr=22050``).
    To preserve the native sampling rate of the file, use ``sr=None``.
    Parameters
    ----------
    path : string, int, pathlib.Path or file-like object
        path to the input file.
        Any codec supported by `soundfile` or `audioread` will work.
        Any string file paths, or any object implementing Python's
        file interface (e.g. `pathlib.Path`) are supported as `path`.
        If the codec is supported by `soundfile`, then `path` can also be
        an open file descriptor (int).
    sr   : number > 0 [scalar]
        target sampling rate
        'None' uses the native sampling rate
    mono : bool
        convert signal to mono
    offset : float
        start reading after this time (in seconds)
    duration : float
        only load up to this much audio (in seconds)
    dtype : numeric type
        data type of ``y``
    res_type : str
        resample type (see note)
        .. note::
            By default, this uses `resampy`'s high-quality mode ('kaiser_best').
            For alternative resampling modes, see `resample`
        .. note::
           `audioread` may truncate the precision of the audio data to 16 bits.
           See :ref:`ioformats` for alternate loading methods.
    Returns
    -------
    y    : np.ndarray [shape=(n,) or (2, n)]
        audio time series
    sr   : number > 0 [scalar]
        sampling rate of ``y``
    Examples
    --------
    >>> # Load an ogg vorbis file
    >>> filename = librosa.ex('trumpet')
    >>> y, sr = librosa.load(filename)
    >>> y
    array([-1.407e-03, -4.461e-04, ..., -3.042e-05,  1.277e-05],
          dtype=float32)
    >>> sr
    22050
    >>> # Load a file and resample to 11 KHz
    >>> filename = librosa.ex('trumpet')
    >>> y, sr = librosa.load(filename, sr=11025)
    >>> y
    array([-8.746e-04, -3.363e-04, ..., -1.301e-05,  0.000e+00],
          dtype=float32)
    >>> sr
    11025
    >>> # Load 5 seconds of a file, starting 15 seconds in
    >>> filename = librosa.ex('brahms')
    >>> y, sr = librosa.load(filename, offset=15.0, duration=5.0)
    >>> y
    array([0.146, 0.144, ..., 0.128, 0.015], dtype=float32)
    >>> sr
    22050
    """

    # If soundfile failed, try audioread instead
    if isinstance(path, (str, pathlib.PurePath)):
        y, sr_native = __audioread_load(path, offset, duration, dtype)
    else:
        raise (exc)

    # Final cleanup for dtype and contiguity
    if mono:
        y = to_mono(y)

    if sr is not None:
        y = resample(y, sr_native, sr, res_type=res_type)

    else:
        sr = sr_native

    return y, sr


def __audioread_load(path, offset, duration, dtype):
    """Load an audio buffer using audioread.
    This loads one block at a time, and then concatenates the results.
    """

    y = []
    with audioread.audio_open(path) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels

        s_start = int(np.round(sr_native * offset)) * n_channels

        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + (int(np.round(sr_native * duration)) * n_channels)

        n = 0

        for frame in input_file:
            frame = util.buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)

            if n < s_start:
                # offset is after the current frame
                # keep reading
                continue

            if s_end < n_prev:
                # we're off the end.  stop reading
                break

            if s_end < n:
                # the end is in this frame.  crop.
                frame = frame[: s_end - n_prev]

            if n_prev <= s_start <= n:
                # beginning is in this frame
                frame = frame[(s_start - n_prev) :]

            # tack on the current frame
            y.append(frame)

    if y:
        y = np.concatenate(y)
        if n_channels > 1:
            y = y.reshape((-1, n_channels)).T
    else:
        y = np.empty(0, dtype=dtype)

    return y, sr_native


#@cache(level=20)
def to_mono(y):
    """Convert an audio signal to mono by averaging samples across channels.
    Parameters
    ----------
    y : np.ndarray [shape=(2,n) or shape=(n,)]
        audio time series, either stereo or mono
    Returns
    -------
    y_mono : np.ndarray [shape=(n,)]
        ``y`` as a monophonic time-series
    Notes
    -----
    This function caches at level 20.
    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet', hq=True), mono=False)
    >>> y.shape
    (2, 117601)
    >>> y_mono = librosa.to_mono(y)
    >>> y_mono.shape
    (117601,)
    """
    # Ensure Fortran contiguity.
    y = np.asfortranarray(y)

    # Validate the buffer.  Stereo is ok here.
    util.valid_audio(y, mono=False)

    if y.ndim > 1:
        y = np.mean(y, axis=0)

    return y


def resample(
    y, orig_sr, target_sr, res_type="kaiser_best", fix=True, scale=False, **kwargs
):
    """Resample a time series from orig_sr to target_sr
    By default, this uses a high-quality (but relatively slow) method ('kaiser_best')
    for band-limited sinc interpolation.  The alternate ``res_type`` values listed below
    offer different trade-offs of speed and quality.
    Parameters
    ----------
    y : np.ndarray [shape=(n,) or shape=(2, n)]
        audio time series.  Can be mono or stereo.
    orig_sr : number > 0 [scalar]
        original sampling rate of ``y``
    target_sr : number > 0 [scalar]
        target sampling rate
    res_type : str
        resample type (see note)
        .. note::
            By default, this uses `resampy`'s high-quality mode ('kaiser_best').
            To use a faster method, set ``res_type='kaiser_fast'``.
            To use `scipy.signal.resample`, set ``res_type='fft'`` or ``res_type='scipy'``. (slow)
            To use `scipy.signal.resample_poly`, set ``res_type='polyphase'``. (fast)
            To use `samplerate.converters.resample`, set any of the following:
                - ``res_type='linear'``: linear interpolation (fast)
                - ``res_type='zero_order_hold'``: repeat the last value between samples (very fast)
                - ``res_type='sinc_best'``, ``'sinc_medium'``, or ``'sinc_fastest'``: for high-, medium-,
                  and low-quality sinc interpolation
        .. note::
            When using ``res_type='polyphase'``, only integer sampling rates are
            supported.
    fix : bool
        adjust the length of the resampled signal to be of size exactly
        ``ceil(target_sr * len(y) / orig_sr)``
    scale : bool
        Scale the resampled signal so that ``y`` and ``y_hat`` have approximately
        equal total energy.
    kwargs : additional keyword arguments
        If ``fix==True``, additional keyword arguments to pass to
        `librosa.util.fix_length`.
    Returns
    -------
    y_hat : np.ndarray [shape=(n * target_sr / orig_sr,)]
        ``y`` resampled from ``orig_sr`` to ``target_sr``
    Raises
    ------
    ParameterError
        If ``res_type='polyphase'`` and ``orig_sr`` or ``target_sr`` are not both
        integer-valued.
    See Also
    --------
    librosa.util.fix_length
    scipy.signal.resample
    resampy
    samplerate.converters.resample
    Notes
    -----
    This function caches at level 20.
    Examples
    --------
    Downsample from 22 KHz to 8 KHz
    >>> y, sr = librosa.load(librosa.ex('trumpet'), sr=22050)
    >>> y_8k = librosa.resample(y, sr, 8000)
    >>> y.shape, y_8k.shape
    ((117601,), (42668,))
    """

    # First, validate the audio buffer
    util.valid_audio(y, mono=False)

    if orig_sr == target_sr:
        return y

    ratio = float(target_sr) / orig_sr

    n_samples = int(np.ceil(y.shape[-1] * ratio))

    if res_type in ("scipy", "fft"):
        y_hat = scipy.signal.resample(y, n_samples, axis=-1)
    elif res_type == "polyphase":
        if int(orig_sr) != orig_sr or int(target_sr) != target_sr:
            raise ParameterError(
                "polyphase resampling is only supported for integer-valued sampling rates."
            )

        # For polyphase resampling, we need up- and down-sampling ratios
        # We can get those from the greatest common divisor of the rates
        # as long as the rates are integrable
        orig_sr = int(orig_sr)
        target_sr = int(target_sr)
        gcd = np.gcd(orig_sr, target_sr)
        y_hat = scipy.signal.resample_poly(y, target_sr // gcd, orig_sr // gcd, axis=-1)
    elif res_type in (
        "linear",
        "zero_order_hold",
        "sinc_best",
        "sinc_fastest",
        "sinc_medium",
    ):
        import samplerate

        # We have to transpose here to match libsamplerate
        y_hat = samplerate.resample(y.T, ratio, converter_type=res_type).T
    else:
        y_hat = resampy.resample(y, orig_sr, target_sr, filter=res_type, axis=-1)

    if fix:
        y_hat = util.fix_length(y_hat, n_samples, **kwargs)

    if scale:
        y_hat /= np.sqrt(ratio)

    return np.asfortranarray(y_hat, dtype=y.dtype)


#@cache(level=30)
def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
    """Convert a power spectrogram (amplitude squared) to decibel (dB) units
    This computes the scaling ``10 * log10(S / ref)`` in a numerically
    stable way.
    Parameters
    ----------
    S : np.ndarray
        input power
    ref : scalar or callable
        If scalar, the amplitude ``abs(S)`` is scaled relative to ``ref``::
            10 * log10(S / ref)
        Zeros in the output correspond to positions where ``S == ref``.
        If callable, the reference value is computed as ``ref(S)``.
    amin : float > 0 [scalar]
        minimum threshold for ``abs(S)`` and ``ref``
    top_db : float >= 0 [scalar]
        threshold the output at ``top_db`` below the peak:
        ``max(10 * log10(S)) - top_db``
    Returns
    -------
    S_db : np.ndarray
        ``S_db ~= 10 * log10(S) - 10 * log10(ref)``
    See Also
    --------
    perceptual_weighting
    db_to_power
    amplitude_to_db
    db_to_amplitude
    Notes
    -----
    This function caches at level 30.
    Examples
    --------
    Get a power spectrogram from a waveform ``y``
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> S = np.abs(librosa.stft(y))
    >>> librosa.power_to_db(S**2)
    array([[-41.809, -41.809, ..., -41.809, -41.809],
           [-41.809, -41.809, ..., -41.809, -41.809],
           ...,
           [-41.809, -41.809, ..., -41.809, -41.809],
           [-41.809, -41.809, ..., -41.809, -41.809]], dtype=float32)
    Compute dB relative to peak power
    >>> librosa.power_to_db(S**2, ref=np.max)
    array([[-80., -80., ..., -80., -80.],
           [-80., -80., ..., -80., -80.],
           ...,
           [-80., -80., ..., -80., -80.],
           [-80., -80., ..., -80., -80.]], dtype=float32)
    Or compare to median power
    >>> librosa.power_to_db(S**2, ref=np.median)
    array([[16.578, 16.578, ..., 16.578, 16.578],
           [16.578, 16.578, ..., 16.578, 16.578],
           ...,
           [16.578, 16.578, ..., 16.578, 16.578],
           [16.578, 16.578, ..., 16.578, 16.578]], dtype=float32)
    And plot the results
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> imgpow = librosa.display.specshow(S**2, sr=sr, y_axis='log', x_axis='time',
    ...                                   ax=ax[0])
    >>> ax[0].set(title='Power spectrogram')
    >>> ax[0].label_outer()
    >>> imgdb = librosa.display.specshow(librosa.power_to_db(S**2, ref=np.max),
    ...                                  sr=sr, y_axis='log', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='Log-Power spectrogram')
    >>> fig.colorbar(imgpow, ax=ax[0])
    >>> fig.colorbar(imgdb, ax=ax[1], format="%+2.0f dB")
    """

    S = np.asarray(S)

    if amin <= 0:
        raise ParameterError("amin must be strictly positive")

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn(
            "power_to_db was called on complex input so phase "
            "information will be discarded. To suppress this warning, "
            "call power_to_db(np.abs(D)**2) instead."
        )
        magnitude = np.abs(S)
    else:
        magnitude = S

    if callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    if top_db is not None:
        if top_db < 0:
            raise ParameterError("top_db must be non-negative")
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


def melspectrogram(
    y=None,
    sr=22050,
    S=None,
    n_fft=2048,
    hop_length=512,
    win_length=None,
    window="hann",
    center=True,
    pad_mode="reflect",
    power=2.0,
    **kwargs,
):
    """Compute a mel-scaled spectrogram.
    If a spectrogram input ``S`` is provided, then it is mapped directly onto
    the mel basis by ``mel_f.dot(S)``.
    If a time-series input ``y, sr`` is provided, then its magnitude spectrogram
    ``S`` is first computed, and then mapped onto the mel scale by
    ``mel_f.dot(S**power)``.
    By default, ``power=2`` operates on a power spectrum.
    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time-series
    sr : number > 0 [scalar]
        sampling rate of ``y``
    S : np.ndarray [shape=(d, t)]
        spectrogram
    n_fft : int > 0 [scalar]
        length of the FFT window
    hop_length : int > 0 [scalar]
        number of samples between successive frames.
        See `librosa.stft`
    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()`.
        The window will be of length `win_length` and then padded
        with zeros to match ``n_fft``.
        If unspecified, defaults to ``win_length = n_fft``.
    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a vector or array of length ``n_fft``
        .. see also:: `librosa.filters.get_window`
    center : boolean
        - If `True`, the signal ``y`` is padded so that frame
          ``t`` is centered at ``y[t * hop_length]``.
        - If `False`, then frame ``t`` begins at ``y[t * hop_length]``
    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses reflection padding.
    power : float > 0 [scalar]
        Exponent for the magnitude melspectrogram.
        e.g., 1 for energy, 2 for power, etc.
    kwargs : additional keyword arguments
        Mel filter bank parameters.
        See `librosa.filters.mel` for details.
    Returns
    -------
    S : np.ndarray [shape=(n_mels, t)]
        Mel spectrogram
    See Also
    --------
    librosa.filters.mel
        Mel filter bank construction
    librosa.stft
        Short-time Fourier Transform
    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> librosa.feature.melspectrogram(y=y, sr=sr)
    array([[3.837e-06, 1.451e-06, ..., 8.352e-14, 1.296e-11],
           [2.213e-05, 7.866e-06, ..., 8.532e-14, 1.329e-11],
           ...,
           [1.115e-05, 5.192e-06, ..., 3.675e-08, 2.470e-08],
           [6.473e-07, 4.402e-07, ..., 1.794e-08, 2.908e-08]],
          dtype=float32)
    Using a pre-computed power spectrogram would give the same result:
    >>> D = np.abs(librosa.stft(y))**2
    >>> S = librosa.feature.melspectrogram(S=D, sr=sr)
    Display of mel-frequency spectrogram coefficients, with custom
    arguments for mel filterbank construction (default is fmax=sr/2):
    >>> # Passing through arguments to the Mel filters
    >>> S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
    ...                                     fmax=8000)
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> S_dB = librosa.power_to_db(S, ref=np.max)
    >>> img = librosa.display.specshow(S_dB, x_axis='time',
    ...                          y_axis='mel', sr=sr,
    ...                          fmax=8000, ax=ax)
    >>> fig.colorbar(img, ax=ax, format='%+2.0f dB')
    >>> ax.set(title='Mel-frequency spectrogram')
    """

    S, n_fft = _spectrogram(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        power=power,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    # Build a Mel filter
    mel_basis = filters.mel(sr, n_fft, **kwargs)

    return np.dot(mel_basis, S)


def _spectrogram(
    y=None,
    S=None,
    n_fft=2048,
    hop_length=512,
    power=1,
    win_length=None,
    window="hann",
    center=True,
    pad_mode="reflect",
):
    """Helper function to retrieve a magnitude spectrogram.
    This is primarily used in feature extraction functions that can operate on
    either audio time-series or spectrogram input.
    Parameters
    ----------
    y : None or np.ndarray [ndim=1]
        If provided, an audio time series
    S : None or np.ndarray
        Spectrogram input, optional
    n_fft : int > 0
        STFT window size
    hop_length : int > 0
        STFT hop length
    power : float > 0
        Exponent for the magnitude spectrogram,
        e.g., 1 for energy, 2 for power, etc.
    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by ``window``.
        The window will be of length ``win_length`` and then padded
        with zeros to match ``n_fft``.
        If unspecified, defaults to ``win_length = n_fft``.
    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a vector or array of length ``n_fft``
        .. see also:: `filters.get_window`
    center : boolean
        - If ``True``, the signal ``y`` is padded so that frame
          ``t`` is centered at ``y[t * hop_length]``.
        - If ``False``, then frame ``t`` begins at ``y[t * hop_length]``
    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses reflection padding.
    Returns
    -------
    S_out : np.ndarray [dtype=np.float32]
        - If ``S`` is provided as input, then ``S_out == S``
        - Else, ``S_out = |stft(y, ...)|**power``
    n_fft : int > 0
        - If ``S`` is provided, then ``n_fft`` is inferred from ``S``
        - Else, copied from input
    """

    if S is not None:
        # Infer n_fft from spectrogram shape
        n_fft = 2 * (S.shape[0] - 1)
    else:
        # Otherwise, compute a magnitude spectrogram from input
        S = (
            np.abs(
                stft(
                    y,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    center=center,
                    window=window,
                    pad_mode=pad_mode,
                )
            )
            ** power
        )

    return S, n_fft


#@cache(level=20)
def stft(
    y,
    n_fft=2048,
    hop_length=None,
    win_length=None,
    window="hann",
    center=True,
    dtype=None,
    pad_mode="reflect",
):
    """Short-time Fourier transform (STFT).
    The STFT represents a signal in the time-frequency domain by
    computing discrete Fourier transforms (DFT) over short overlapping
    windows.
    This function returns a complex-valued matrix D such that
    - ``np.abs(D[f, t])`` is the magnitude of frequency bin ``f``
      at frame ``t``, and
    - ``np.angle(D[f, t])`` is the phase of frequency bin ``f``
      at frame ``t``.
    The integers ``t`` and ``f`` can be converted to physical units by means
    of the utility functions `frames_to_sample` and `fft_frequencies`.
    Parameters
    ----------
    y : np.ndarray [shape=(n,)], real-valued
        input signal
    n_fft : int > 0 [scalar]
        length of the windowed signal after padding with zeros.
        The number of rows in the STFT matrix ``D`` is ``(1 + n_fft/2)``.
        The default value, ``n_fft=2048`` samples, corresponds to a physical
        duration of 93 milliseconds at a sample rate of 22050 Hz, i.e. the
        default sample rate in librosa. This value is well adapted for music
        signals. However, in speech processing, the recommended value is 512,
        corresponding to 23 milliseconds at a sample rate of 22050 Hz.
        In any case, we recommend setting ``n_fft`` to a power of two for
        optimizing the speed of the fast Fourier transform (FFT) algorithm.
    hop_length : int > 0 [scalar]
        number of audio samples between adjacent STFT columns.
        Smaller values increase the number of columns in ``D`` without
        affecting the frequency resolution of the STFT.
        If unspecified, defaults to ``win_length // 4`` (see below).
    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by ``window`` of length ``win_length``
        and then padded with zeros to match ``n_fft``.
        Smaller values improve the temporal resolution of the STFT (i.e. the
        ability to discriminate impulses that are closely spaced in time)
        at the expense of frequency resolution (i.e. the ability to discriminate
        pure tones that are closely spaced in frequency). This effect is known
        as the time-frequency localization trade-off and needs to be adjusted
        according to the properties of the input signal ``y``.
        If unspecified, defaults to ``win_length = n_fft``.
    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        Either:
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a vector or array of length ``n_fft``
        Defaults to a raised cosine window (`'hann'`), which is adequate for
        most applications in audio signal processing.
        .. see also:: `filters.get_window`
    center : boolean
        If ``True``, the signal ``y`` is padded so that frame
        ``D[:, t]`` is centered at ``y[t * hop_length]``.
        If ``False``, then ``D[:, t]`` begins at ``y[t * hop_length]``.
        Defaults to ``True``,  which simplifies the alignment of ``D`` onto a
        time grid by means of `librosa.frames_to_samples`.
        Note, however, that ``center`` must be set to `False` when analyzing
        signals with `librosa.stream`.
        .. see also:: `librosa.stream`
    dtype : np.dtype, optional
        Complex numeric type for ``D``.  Default is inferred to match the
        precision of the input signal.
    pad_mode : string or function
        If ``center=True``, this argument is passed to `np.pad` for padding
        the edges of the signal ``y``. By default (``pad_mode="reflect"``),
        ``y`` is padded on both sides with its own reflection, mirrored around
        its first and last sample respectively.
        If ``center=False``,  this argument is ignored.
        .. see also:: `numpy.pad`
    Returns
    -------
    D : np.ndarray [shape=(1 + n_fft/2, n_frames), dtype=dtype]
        Complex-valued matrix of short-term Fourier transform
        coefficients.
    See Also
    --------
    istft : Inverse STFT
    reassigned_spectrogram : Time-frequency reassigned spectrogram
    Notes
    -----
    This function caches at level 20.
    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> S = np.abs(librosa.stft(y))
    >>> S
    array([[5.395e-03, 3.332e-03, ..., 9.862e-07, 1.201e-05],
           [3.244e-03, 2.690e-03, ..., 9.536e-07, 1.201e-05],
           ...,
           [7.523e-05, 3.722e-05, ..., 1.188e-04, 1.031e-03],
           [7.640e-05, 3.944e-05, ..., 5.180e-04, 1.346e-03]],
          dtype=float32)
    Use left-aligned frames, instead of centered frames
    >>> S_left = librosa.stft(y, center=False)
    Use a shorter hop length
    >>> D_short = librosa.stft(y, hop_length=64)
    Display a spectrogram
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> img = librosa.display.specshow(librosa.amplitude_to_db(S,
    ...                                                        ref=np.max),
    ...                                y_axis='log', x_axis='time', ax=ax)
    >>> ax.set_title('Power spectrogram')
    >>> fig.colorbar(img, ax=ax, format="%+2.0f dB")
    """

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    fft_window = get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = util.pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Check audio is valid
    util.valid_audio(y)

    # Pad the time series so that frames are centered
    if center:
        if n_fft > y.shape[-1]:
            warnings.warn(
                "n_fft={} is too small for input signal of length={}".format(
                    n_fft, y.shape[-1]
                )
            )

        y = np.pad(y, int(n_fft // 2), mode=pad_mode)

    elif n_fft > y.shape[-1]:
        raise ParameterError(
            "n_fft={} is too small for input signal of length={}".format(
                n_fft, y.shape[-1]
            )
        )

    # Window the time series.
    y_frames = util.frame(y, frame_length=n_fft, hop_length=hop_length)

    if dtype is None:
        dtype = util.dtype_r2c(y.dtype)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty(
        (int(1 + n_fft // 2), y_frames.shape[1]), dtype=dtype, order="F"
    )

    fft = get_fftlib()

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = util.MAX_MEM_BLOCK // (stft_matrix.shape[0] * stft_matrix.itemsize)
    n_columns = max(n_columns, 1)

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        stft_matrix[:, bl_s:bl_t] = fft.rfft(
            fft_window * y_frames[:, bl_s:bl_t], axis=0
        )
    return stft_matrix


def get_fftlib():
    """Get the FFT library currently used by librosa
    Returns
    -------
    fft : module
        The FFT library currently used by librosa.
        Must API-compatible with `numpy.fft`.
    """
    from numpy import fft

    return fft


#@cache(level=10)
def get_window(window, Nx, fftbins=True):
    """Compute a window function.
    This is a wrapper for `scipy.signal.get_window` that additionally
    supports callable or pre-computed windows.
    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        The window specification:
        - If string, it's the name of the window function (e.g., `'hann'`)
        - If tuple, it's the name of the window function and any parameters
          (e.g., `('kaiser', 4.0)`)
        - If numeric, it is treated as the beta parameter of the `'kaiser'`
          window, as in `scipy.signal.get_window`.
        - If callable, it's a function that accepts one integer argument
          (the window length)
        - If list-like, it's a pre-computed window of the correct length `Nx`
    Nx : int > 0
        The length of the window
    fftbins : bool, optional
        If True (default), create a periodic window for use with FFT
        If False, create a symmetric window for filter design applications.
    Returns
    -------
    get_window : np.ndarray
        A window of length `Nx` and type `window`
    See Also
    --------
    scipy.signal.get_window
    Notes
    -----
    This function caches at level 10.
    Raises
    ------
    ParameterError
        If `window` is supplied as a vector of length != `n_fft`,
        or is otherwise mis-specified.
    """
    if callable(window):
        return window(Nx)

    elif isinstance(window, (str, tuple)) or np.isscalar(window):
        # TODO: if we add custom window functions in librosa, call them here

        return scipy.signal.get_window(window, Nx, fftbins=fftbins)

    elif isinstance(window, (np.ndarray, list)):
        if len(window) == Nx:
            return np.asarray(window)

        raise ParameterError(
            "Window size mismatch: " "{:d} != {:d}".format(len(window), Nx)
        )
    else:
        raise ParameterError("Invalid window specification: {}".format(window))