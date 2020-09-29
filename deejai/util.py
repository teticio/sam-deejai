import numpy as np
from numpy.lib.stride_tricks import as_strided

# Constrain STFT block sizes to 256 KB
MAX_MEM_BLOCK = 2 ** 8 * 2 ** 10


def buf_to_float(x, n_bytes=2, dtype=np.float32):
    """Convert an integer buffer to floating point values.
    This is primarily useful when loading integer-valued wav data
    into numpy arrays.
    Parameters
    ----------
    x : np.ndarray [dtype=int]
        The integer-valued data buffer
    n_bytes : int [1, 2, 4]
        The number of bytes per sample in ``x``
    dtype : numeric type
        The target output type (default: 32-bit float)
    Returns
    -------
    x_float : np.ndarray [dtype=float]
        The input data buffer cast to floating point
    """

    # Invert the scale of the data
    scale = 1.0 / float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = "<i{:d}".format(n_bytes)

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)


#@cache(level=20)
def valid_audio(y, mono=True):
    """Determine whether a variable contains valid audio data.
    If ``mono=True``, then ``y`` is only considered valid if it has shape
    ``(N,)`` (number of samples).
    If ``mono=False``, then ``y`` may be either monophonic, or have shape
    ``(2, N)`` (stereo) or ``(K, N)`` for ``K>=2`` for general multi-channel.
    Parameters
    ----------
    y : np.ndarray
      The input data to validate
    mono : bool
      Whether or not to require monophonic audio
    Returns
    -------
    valid : bool
        True if all tests pass
    Raises
    ------
    ParameterError
        In any of these cases:
            - ``type(y)`` is not ``np.ndarray``
            - ``y.dtype`` is not floating-point
            - ``mono == True`` and ``y.ndim`` is not 1
            - ``mono == False`` and ``y.ndim`` is not 1 or 2
            - ``mono == False`` and ``y.ndim == 2`` but ``y.shape[0] == 1``
            - ``np.isfinite(y).all()`` is False
    Notes
    -----
    This function caches at level 20.
    Examples
    --------
    >>> # By default, valid_audio allows only mono signals
    >>> filepath = librosa.ex('trumpet', hq=True)
    >>> y_mono, sr = librosa.load(filepath, mono=True)
    >>> y_stereo, _ = librosa.load(filepath, mono=False)
    >>> librosa.util.valid_audio(y_mono), librosa.util.valid_audio(y_stereo)
    True, False
    >>> # To allow stereo signals, set mono=False
    >>> librosa.util.valid_audio(y_stereo, mono=False)
    True
    See also
    --------
    numpy.float32
    """

    if not isinstance(y, np.ndarray):
        raise ParameterError("Audio data must be of type numpy.ndarray")

    if not np.issubdtype(y.dtype, np.floating):
        raise ParameterError("Audio data must be floating-point")

    if mono and y.ndim != 1:
        raise ParameterError(
            "Invalid shape for monophonic audio: "
            "ndim={:d}, shape={}".format(y.ndim, y.shape)
        )

    elif y.ndim > 2 or y.ndim == 0:
        raise ParameterError(
            "Audio data must have shape (samples,) or (channels, samples). "
            "Received shape={}".format(y.shape)
        )

    elif y.ndim == 2 and y.shape[0] < 2:
        raise ParameterError(
            "Mono data must have shape (samples,). " "Received shape={}".format(y.shape)
        )

    if not np.isfinite(y).all():
        raise ParameterError("Audio buffer is not finite everywhere")

    return True


def fix_length(data, size, axis=-1, **kwargs):
    """Fix the length an array ``data`` to exactly ``size`` along a target axis.
    If ``data.shape[axis] < n``, pad according to the provided kwargs.
    By default, ``data`` is padded with trailing zeros.
    Examples
    --------
    >>> y = np.arange(7)
    >>> # Default: pad with zeros
    >>> librosa.util.fix_length(y, 10)
    array([0, 1, 2, 3, 4, 5, 6, 0, 0, 0])
    >>> # Trim to a desired length
    >>> librosa.util.fix_length(y, 5)
    array([0, 1, 2, 3, 4])
    >>> # Use edge-padding instead of zeros
    >>> librosa.util.fix_length(y, 10, mode='edge')
    array([0, 1, 2, 3, 4, 5, 6, 6, 6, 6])
    Parameters
    ----------
    data : np.ndarray
      array to be length-adjusted
    size : int >= 0 [scalar]
      desired length of the array
    axis : int, <= data.ndim
      axis along which to fix length
    kwargs : additional keyword arguments
        Parameters to ``np.pad``
    Returns
    -------
    data_fixed : np.ndarray [shape=data.shape]
        ``data`` either trimmed or padded to length ``size``
        along the specified axis.
    See Also
    --------
    numpy.pad
    """

    kwargs.setdefault("mode", "constant")

    n = data.shape[axis]

    if n > size:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, size)
        return data[tuple(slices)]

    elif n < size:
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (0, size - n)
        return np.pad(data, lengths, **kwargs)

    return data


def pad_center(data, size, axis=-1, **kwargs):
    """Pad an array to a target length along a target axis.
    This differs from `np.pad` by centering the data prior to padding,
    analogous to `str.center`
    Examples
    --------
    >>> # Generate a vector
    >>> data = np.ones(5)
    >>> librosa.util.pad_center(data, 10, mode='constant')
    array([ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.])
    >>> # Pad a matrix along its first dimension
    >>> data = np.ones((3, 5))
    >>> librosa.util.pad_center(data, 7, axis=0)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> # Or its second dimension
    >>> librosa.util.pad_center(data, 7, axis=1)
    array([[ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.]])
    Parameters
    ----------
    data : np.ndarray
        Vector to be padded and centered
    size : int >= len(data) [scalar]
        Length to pad ``data``
    axis : int
        Axis along which to pad and center the data
    kwargs : additional keyword arguments
      arguments passed to `np.pad`
    Returns
    -------
    data_padded : np.ndarray
        ``data`` centered and padded to length ``size`` along the
        specified axis
    Raises
    ------
    ParameterError
        If ``size < data.shape[axis]``
    See Also
    --------
    numpy.pad
    """

    kwargs.setdefault("mode", "constant")

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise ParameterError(
            ("Target size ({:d}) must be " "at least input size ({:d})").format(size, n)
        )

    return np.pad(data, lengths, **kwargs)


def frame(x, frame_length, hop_length, axis=-1):
    """Slice a data array into (overlapping) frames.
    This implementation uses low-level stride manipulation to avoid
    making a copy of the data.  The resulting frame representation
    is a new view of the same input data.
    However, if the input data is not contiguous in memory, a warning
    will be issued and the output will be a full copy, rather than
    a view of the input data.
    For example, a one-dimensional input ``x = [0, 1, 2, 3, 4, 5, 6]``
    can be framed with frame length 3 and hop length 2 in two ways.
    The first (``axis=-1``), results in the array ``x_frames``::
        [[0, 2, 4],
         [1, 3, 5],
         [2, 4, 6]]
    where each column ``x_frames[:, i]`` contains a contiguous slice of
    the input ``x[i * hop_length : i * hop_length + frame_length]``.
    The second way (``axis=0``) results in the array ``x_frames``::
        [[0, 1, 2],
         [2, 3, 4],
         [4, 5, 6]]
    where each row ``x_frames[i]`` contains a contiguous slice of the input.
    This generalizes to higher dimensional inputs, as shown in the examples below.
    In general, the framing operation increments by 1 the number of dimensions,
    adding a new "frame axis" either to the end of the array (``axis=-1``)
    or the beginning of the array (``axis=0``).
    Parameters
    ----------
    x : np.ndarray
        Array to frame
    frame_length : int > 0 [scalar]
        Length of the frame
    hop_length : int > 0 [scalar]
        Number of steps to advance between frames
    axis : 0 or -1
        The axis along which to frame.
        If ``axis=-1`` (the default), then ``x`` is framed along its last dimension.
        ``x`` must be "F-contiguous" in this case.
        If ``axis=0``, then ``x`` is framed along its first dimension.
        ``x`` must be "C-contiguous" in this case.
    Returns
    -------
    x_frames : np.ndarray [shape=(..., frame_length, N_FRAMES) or (N_FRAMES, frame_length, ...)]
        A framed view of ``x``, for example with ``axis=-1`` (framing on the last dimension)::
            x_frames[..., j] == x[..., j * hop_length : j * hop_length + frame_length]
        If ``axis=0`` (framing on the first dimension), then::
            x_frames[j] = x[j * hop_length : j * hop_length + frame_length]
    Raises
    ------
    ParameterError
        If ``x`` is not an `np.ndarray`.
        If ``x.shape[axis] < frame_length``, there is not enough data to fill one frame.
        If ``hop_length < 1``, frames cannot advance.
        If ``axis`` is not 0 or -1.  Framing is only supported along the first or last axis.
    See Also
    --------
    numpy.asfortranarray : Convert data to F-contiguous representation
    numpy.ascontiguousarray : Convert data to C-contiguous representation
    numpy.ndarray.flags : information about the memory layout of a numpy `ndarray`.
    Examples
    --------
    Extract 2048-sample frames from monophonic signal with a hop of 64 samples per frame
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> frames = librosa.util.frame(y, frame_length=2048, hop_length=64)
    >>> frames
    array([[-1.407e-03, -2.604e-02, ..., -1.795e-05, -8.108e-06],
           [-4.461e-04, -3.721e-02, ..., -1.573e-05, -1.652e-05],
           ...,
           [ 7.960e-02, -2.335e-01, ..., -6.815e-06,  1.266e-05],
           [ 9.568e-02, -1.252e-01, ...,  7.397e-06, -1.921e-05]],
          dtype=float32)
    >>> y.shape
    (117601,)
    >>> frames.shape
    (2048, 1806)
    Or frame along the first axis instead of the last:
    >>> frames = librosa.util.frame(y, frame_length=2048, hop_length=64, axis=0)
    >>> frames.shape
    (1806, 2048)
    Frame a stereo signal:
    >>> y, sr = librosa.load(librosa.ex('trumpet', hq=True), mono=False)
    >>> y.shape
    (2, 117601)
    >>> frames = librosa.util.frame(y, frame_length=2048, hop_length=64)
    (2, 2048, 1806)
    Carve an STFT into fixed-length patches of 32 frames with 50% overlap
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> S = np.abs(librosa.stft(y))
    >>> S.shape
    (1025, 230)
    >>> S_patch = librosa.util.frame(S, frame_length=32, hop_length=16)
    >>> S_patch.shape
    (1025, 32, 13)
    >>> # The first patch contains the first 32 frames of S
    >>> np.allclose(S_patch[:, :, 0], S[:, :32])
    True
    >>> # The second patch contains frames 16 to 16+32=48, and so on
    >>> np.allclose(S_patch[:, :, 1], S[:, 16:48])
    True
    """

    if not isinstance(x, np.ndarray):
        raise ParameterError(
            "Input must be of type numpy.ndarray, " "given type(x)={}".format(type(x))
        )

    if x.shape[axis] < frame_length:
        raise ParameterError(
            "Input is too short (n={:d})"
            " for frame_length={:d}".format(x.shape[axis], frame_length)
        )

    if hop_length < 1:
        raise ParameterError("Invalid hop_length: {:d}".format(hop_length))

    if axis == -1 and not x.flags["F_CONTIGUOUS"]:
        warnings.warn(
            "librosa.util.frame called with axis={} "
            "on a non-contiguous input. This will result in a copy.".format(axis)
        )
        x = np.asfortranarray(x)
    elif axis == 0 and not x.flags["C_CONTIGUOUS"]:
        warnings.warn(
            "librosa.util.frame called with axis={} "
            "on a non-contiguous input. This will result in a copy.".format(axis)
        )
        x = np.ascontiguousarray(x)

    n_frames = 1 + (x.shape[axis] - frame_length) // hop_length
    strides = np.asarray(x.strides)

    new_stride = np.prod(strides[strides > 0] // x.itemsize) * x.itemsize

    if axis == -1:
        shape = list(x.shape)[:-1] + [frame_length, n_frames]
        strides = list(strides) + [hop_length * new_stride]

    elif axis == 0:
        shape = [n_frames, frame_length] + list(x.shape)[1:]
        strides = [hop_length * new_stride] + list(strides)

    else:
        raise ParameterError("Frame axis={} must be either 0 or -1".format(axis))

    return as_strided(x, shape=shape, strides=strides)


def dtype_r2c(d, default=np.complex64):
    """Find the complex numpy dtype corresponding to a real dtype.
    This is used to maintain numerical precision and memory footprint
    when constructing complex arrays from real-valued data
    (e.g. in a Fourier transform).
    A `float32` (single-precision) type maps to `complex64`,
    while a `float64` (double-precision) maps to `complex128`.
    Parameters
    ----------
    d : np.dtype
        The real-valued dtype to convert to complex.
        If ``d`` is a complex type already, it will be returned.
    default : np.dtype, optional
        The default complex target type, if ``d`` does not match a
        known dtype
    Returns
    -------
    d_c : np.dtype
        The complex dtype
    See Also
    --------
    dtype_c2r
    numpy.dtype
    Examples
    --------
    >>> librosa.util.dtype_r2c(np.float32)
    dtype('complex64')
    >>> librosa.util.dtype_r2c(np.int16)
    dtype('complex64')
    >>> librosa.util.dtype_r2c(np.complex128)
    dtype('complex128')
    """
    mapping = {
        np.dtype(np.float32): np.complex64,
        np.dtype(np.float64): np.complex128,
        np.dtype(np.float): np.complex,
    }

    # If we're given a complex type already, return it
    dt = np.dtype(d)
    if dt.kind == "c":
        return dt

    # Otherwise, try to map the dtype.
    # If no match is found, return the default.
    return np.dtype(mapping.get(dt, default))