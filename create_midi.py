import numpy as np

import sys

from mingus.containers import Note
from mingus.midi import fluidsynth
from mingus.containers import Bar, Track, Composition
from mingus.midi import midi_file_out


def main(argv):
    if (argv == None or len(argv) != 1):
        print("Usage: python3 create_midi.py *path/to/output.txt*")
        sys.exit()
    # replace models directory when running from shell script create_mp3.sh
    model_name = argv[0].replace("models/", "")
    model_name = model_name.replace("/output.txt", "")

    print(model_name)
    # initialize soundfont file that will be used to play music
    # fluidsynth.init("soundfonts/YamahaC5Grand-v2_4.sf2", "oss")
    # fluidsynth.init(
    #     "soundfonts/040_Florestan_String_Quartet.sf2", "coreaudio")
    fluidsynth.init("soundfonts/Yamaha_C5_Grand-v2.4.sf2", "coreaudio")
    # fluidsynth.init("soundfonts/040_Florestan_String_Quartet.sf2", "alsa")

    # load original music
    # original = np.loadtxt("input/contrapunctusXIV.txt")
    original = []

    # load network output music
    # output = np.loadtxt("input.txt")
    output = np.loadtxt("output/" + model_name + "/output.txt")

    # concatenate
    complete = np.concatenate((output, output), axis=0)

    # create midi for output and comlete (original + output)
    # for run, voices in enumerate([output]):
    for run, voices in enumerate([output, complete]):
        # create 4 tracks for the 4 voices
        encoded_voices = [Track(), Track(), Track(), Track()]

        # loop through the generated voices
        for i, notes in enumerate([voices[:, 0], voices[:, 1], voices[:, 2], voices[:, 3]]):
            # initialize as impossible note
            last_note = -1
            count = 1
            for j, note in enumerate(notes):
                if note:
                    if ((note == last_note) or (j == 0)):
                        # same note as previous note
                        count += 1
                        last_note = note

                        if (j + count > len(notes)):
                            # current note reaches end of file
                            n = Note()
                            n.from_int(int(last_note))
                            b = Bar()
                            b.place_notes(n, 16/count)
                            encoded_voices[i].add_bar(b)
                    else:
                        # different note encountered
                        # add previous note with its duration to track
                        n = Note()
                        n.from_int(int(last_note))
                        b = Bar()

                        # 8 should be 1/2 -> 2
                        # 16 should be 1 -> 1
                        # 32 should be 2 -> 0.5
                        b.place_notes(n, duration=16/count)
                        encoded_voices[i].add_bar(b)

                        # reset
                        count = 1
                        last_note = note
                else:
                    # current note = 0, means a pause (silence)
                    b = Bar()
                    b.place_rest(16)
                    encoded_voices[i].add_bar(b)

        output_composition = Composition()
        output_composition.add_track(encoded_voices[0])
        output_composition.add_track(encoded_voices[1])
        output_composition.add_track(encoded_voices[2])
        output_composition.add_track(encoded_voices[3])

        if (run == 0):
            midi_file_out.write_Composition(
                "output/output.midi", output_composition)
        else:
            midi_file_out.write_Composition(
                "output/complete.midi", output_composition)


if __name__ == "__main__":
    main(sys.argv[1:])
