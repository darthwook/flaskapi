
import numpy as np

def peak_accel_threshold(data, timestamps, threshold):
    last_state = 'below' # below or above
    crest_troughs = 0
    crossings =[]

    for i, datum in enumerate(data):

        current_state = last_state
        if datum < threshold:
            current_state = 'below'
        elif datum > threshold:
            current_state = 'above'

        if current_state is not last_state: #state changed
            crossing = [timestamps[i], threshold]
            crossings.append(crossing)
            crest_troughs += 1

        last_state = current_state

    return np.array(crossings)
