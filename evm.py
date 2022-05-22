import cv2 as cv
import numpy as np
from scipy.signal import detrend
import scipy.fftpack as fftpack


def temporal_ideal_filter(arr, low, high, fps, axis=0):
    """ Applies a temporal ideal filter to a numpy array

    Paremeters
    ----------
    - arr: a numpy array with shape (N, H, W, C)
        - N: number of frames
        - H: height
        - W: width
        - C: channels
    - low: the low frequency bound
    - high: the high frequency bound
    - fps: the video frame rate
    - axis: the axis of video, should always be 0

    Returns
    -------
    - the array with the filter applied
    """
    fft = fftpack.fft(arr, axis=axis)
    frequencies = fftpack.fftfreq(arr.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - low)).argmin()
    bound_high = (np.abs(frequencies - high)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    iff=fftpack.ifft(fft, axis=axis)
    return np.abs(iff)


def reconstruct_video_g(amp_video, original_video, levels=3):
    """ reconstructs a video from a gaussian pyramid and the original

    Parameters
    ----------
    - amp_video: the amplified gaussian video
    - original_video: the original video
    - levels: the levels in the gaussian video

    Returns
    --------
    - the reconstructed video
    """
    final_video = np.zeros(original_video.shape)
    for i in range(0, amp_video.shape[0]):
        img = amp_video[i]
        for x in range(levels):
            img = cv.pyrUp(img)
        img = img + original_video[i]
        final_video[i] = img
    return final_video


def build_gaussian_pyramid(src, levels=3):
    """ Builds a gaussian pyramid

    Parameters
    ----------
    - src: the input image
    - levels: the number levels in the gaussian pyramid

    Returns
    -------
    - A gaussian pyramid
    """
    s = src.copy()
    pyramid = [s]
    for i in range(levels):
        s = cv.pyrDown(s)
        pyramid.append(s)
    return pyramid


def gaussian_video(video, levels=3):
    """ generates a gaussian pyramid for each frame in a video

    Parameters
    ----------
    - video: the input video array
    - levels: the number of levels in the gaussian pyramid

    Returns
    -------
    - the gaussian video
    """
    n = video.shape[0]
    for i in range(0, n):
        pyr = build_gaussian_pyramid(video[i], levels=levels)
        gaussian_frame=pyr[-1]
        if i==0:
            vid_data = np.zeros((n, *gaussian_frame.shape))
        vid_data[i] = gaussian_frame
    return vid_data


def find_heart_rate(vid, times, fps, low, high, levels=3, alpha=20):
    """ calculates the heart rate of a given face video

    Parameters
    ----------
    - vid: the video

    """
    res = magnify_color(vid, fps, low, high, levels, alpha)
    num_frames = vid.shape[0]

    true_fps = num_frames / (times[-1] - times[0])

    avg = np.mean(res, axis=(1, 2, 3))
    even_times = np.linspace(times[0], times[-1], num_frames)

    processed = detrend(avg)#detrend the signal to avoid interference of light change
    interpolated = np.interp(even_times, times, processed) #interpolation by 1
    interpolated = np.hamming(num_frames) * interpolated#make the signal become more periodic (advoid spectral leakage)
    norm = interpolated/np.linalg.norm(interpolated)
    raw = np.fft.rfft(norm*30)

    freqs = float(true_fps) / num_frames * np.arange(num_frames / 2 + 1)
    freqs_ = 60. * freqs

    fft = np.abs(raw)**2#get amplitude spectrum

    idx = np.where((freqs_ > 50) & (freqs_ < 180))#the range of frequency that HR is supposed to be within
    pruned = fft[idx]
    pfreq = freqs_[idx]

    freqs = pfreq
    fft = pruned

    idx2 = np.argmax(pruned)#max in the range can be HR

    bpm = freqs[idx2]
    return bpm


def magnify_color(vid, fps, low, high, levels=3, alpha=20):
    """ Magnifies the color of a video

    Parameters
    ----------
    - vid: the input video as a numpy array
    - fps: the frame rate of the video
    - low: the low frequency band to amplify
    - high: the high frequency band to amplify
    - levels: the depth at which to make the gaussian pyramid
    - alpha: the factor with which to amplify the color

    Returns
    -------
    - The video with amplified color
    """
    gauss = gaussian_video(vid, levels=levels)
    filtered = temporal_ideal_filter(gauss, low, high, fps)
    amplified = alpha * filtered
    return reconstruct_video_g(amplified, vid, levels=levels)
