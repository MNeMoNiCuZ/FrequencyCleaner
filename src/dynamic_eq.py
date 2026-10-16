"""Frame-dependent EQ envelopes shared by rendering and the playhead display."""
import numpy as np


def band_levels(bands, profile):
    """Stereo-linked peak level inside a curve, in amplitude dBFS.

    scipy's one-sided STFT has half the sinusoidal amplitude. Power linking
    preserves opposite-phase channels. Shape weighting excludes the shoulders.
    """
    weight = np.abs(profile) / max(float(np.max(np.abs(profile))), 1e-12)
    power = sum(np.abs(b) ** 2 for b in bands) / len(bands)
    amplitude = 2 * np.sqrt(np.max(power * weight[:, None], axis=0))
    return 20 * np.log10(np.maximum(amplitude, 1e-12))


def activity_envelope(levels, curve, maximum, hop_seconds):
    if not curve['dynamic']:
        return np.ones(len(levels), dtype=np.float32)
    reduction = np.maximum(levels - curve['threshold'], 0.) * (1. - 1. / curve['ratio'])
    target = np.clip(reduction / max(maximum, 1e-6), 0., 1.)
    attack = np.exp(-hop_seconds / (curve['attack'] / 1000.)) if curve['attack'] else 0.
    release = np.exp(-hop_seconds / (curve['release'] / 1000.)) if curve['release'] else 0.
    result = np.empty(len(target), dtype=np.float32)
    previous = 0.
    for i, value in enumerate(target):
        coefficient = attack if value > previous else release
        previous = coefficient * previous + (1. - coefficient) * value
        result[i] = previous
    return result


def response_at(trace, sample_frame):
    """Interpolate recorded STFT envelopes at the audio playhead."""
    if not trace or not len(trace.get('activity', [])):
        return None
    activity = trace['activity']
    position = np.clip(sample_frame / trace['hop'], 0, activity.shape[1] - 1)
    i = int(position)
    j = min(i + 1, activity.shape[1] - 1)
    factors = activity[:, i] * (1 - (position - i)) + activity[:, j] * (position - i)
    response = np.sum(trace['profiles'] * factors[:, None], axis=0)
    return trace['freqs'], np.clip(response, -120., 36.), factors
