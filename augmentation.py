import numpy as np

# pitch representation scaled to equal distances between chroma and circle of fifths
def pitchEncoding(midi_note, min_note, max_note):
    # n gives distance in semitones of the midi note to A4 (midi = 69)
    # A4 has frequency 440hz, which we use to calc the note freq
    n = midi_note - 69
    freq = np.power(2, (n / 12)) * 440

    # calc min and max pitch
    min_pitch = 2 * np.log2(np.power(2, ((min_note - 69) / 12)) * 440)
    max_pitch = 2 * np.log2(np.power(2, ((max_note - 69) / 12)) * 440)

    # scale such that distance of 1 octave in the first dimension, 
    #   is equal to opposite sides of the chroma/fifth circles
    encoded_pitch = 2 * np.log2(freq) - max_pitch + (max_pitch - min_pitch)/2

    return encoded_pitch

# function to calculate the chroma circle representation of a note
def chromaCircleEncoding(note):
    # chroma circle note positions are 1 to 12 followed
    chroma = list(range(1, 13))

    # calc circle angle
    theta = (chroma[note] - 1) * 30 # 360/12

    # calc x and y
    x = np.cos(theta)
    y = np.sin(theta)

    return x, y

# Function to calculate the circle of fifths representation of a note
def circleOfFifthsEncoding(note):
    # c of fifths note positions are more chaotic 
    c_of_5 = [1, 8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6]

    # calc circle angle
    theta = (c_of_5[note] - 1) * 30 # 360/12

    # calc x and y
    x = np.cos(theta)
    y = np.sin(theta)

    return x, y

# function to calculate the chroma and circle of fifths representations of a note
def calcEncoding(midi_note, min_note, max_note):
    # midi value of note to actual note
    note = int(((midi_note - 55) % 12))

    chroma_x, chroma_y = chromaCircleEncoding(note)
    fifths_x, fifths_y = circleOfFifthsEncoding(note)
    pitch = pitchEncoding(midi_note, min_note, max_note)

    return [pitch, chroma_x, chroma_y, fifths_x, fifths_y]


def createEncodedData(voices):
    # create new shape of data
    encoded_voices = np.zeros((voices.shape[0], voices.shape[1] * 5), dtype=np.float32)

    # check if there are only zeros in the array (useful when needing to encode small number of steps)
    if (not(np.nonzero(voices))):
        return encoded_voices

    # min, max of all notes, ignoring 0's
    min_note = np.min(voices[np.nonzero(voices)])
    max_note = np.max(voices[np.nonzero(voices)])

    for i, voice in enumerate([voices[:,0], voices[:,1], voices[:,2], voices[:,3]]):
        for j, note in enumerate(voice):
            # if silence
            if (note == 0):                
                encoded_voices[j, i*5:i*5+5] = [0., 0., 0., 0., 0.]
            # else note
            else:
                encoded_voices[j, i*5:i*5+5] = calcEncoding(note, min_note, max_note)

    return encoded_voices