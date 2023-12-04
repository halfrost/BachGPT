#!/bin/bash
if [ $# -ne 1 ] 
then
    echo "Usage: ./create_mp3 *path/to/model.pth*"
else
    python3 bach_inference.py $1 #make sure you change the params in this file (model + model params)
    python3 create_midi.py $1
    # fluidsynth soundfonts/YamahaC5Grand-v2_4.sf2 -F output/output.mp3 output/output.midi
    timidity output/output.midi -Ow -o - | ffmpeg -i - -vcodec libx265 -crf 28 output/output.mp3
    timidity output/complete.midi -Ow -o - | ffmpeg -i - -vcodec libx265 -crf 28 output/complete.mp3
fi