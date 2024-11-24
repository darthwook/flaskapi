import numpy as np

GRAVITY = 9.81


def adaptive_step_jk_threshold(data, timestamps, zero=GRAVITY):
    last_state = None
    current_state = None
    last_peak = None
    last_trough = None

    peak_troughs = []

    jk_mean = 1
    alpha = 0.125

    jk_dev = 0.125
    beta = 0.25

    # Graphing Purpose Array
    # 0 - timestamp
    # 1 - jk Mean
    # 2 - jk Standard Deviation
    meta = []

    for i, datum in enumerate(data):

        if datum < zero and last_state != None:
            current_state = 'trough'
            if last_trough == None or datum < last_trough["val"]:
                last_trough = {
                    "ts": timestamps[i],
                    "val": data[i],
                    "min_max": "min"
                }
        elif datum > zero:
            current_state = 'peak'
            if last_peak == None or datum > last_peak["val"]:
                last_peak = {
                    "ts": timestamps[i],
                    "val": data[i],
                    "min_max": "max"
                }

        if current_state != last_state:
            # Zero Crossing
            # When coming out of trough assess the "Dip"
            if last_state == 'trough':

                if last_peak:

                    jk = last_peak['val'] - last_trough['val']

                    if jk > jk_mean - 4 * jk_dev:
                        jk_dev = abs(jk_mean - jk) * beta + jk_dev * (1 - beta)
                        jk_mean = jk * alpha + jk_mean * (1 - alpha)

                        peak_troughs.append(last_trough)

                        meta.append([
                            timestamps[i],
                            jk_mean,
                            jk_dev
                        ])

                last_trough = None
            elif last_state == 'peak':
                # peak_troughs.append(last_peak)
                last_peak = None

        last_state = current_state

    return np.array(peak_troughs), np.array(meta)
