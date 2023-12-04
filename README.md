# BachGPT


# How to run

I highly recommend running the notebook in [Colaboratory](https://colab.research.google.com/) so you can view the results and change parameters in real-time. The jupyter notebook code is [here](). However, you can run Python code on your local machine.

## 1. Train model

```python
python3 bachGPT.py 
```

## 2. Use the model to generate predicted output

```python
python3 bach_predict.py ./models/model96256.pth 
```

## 3. Create midi music file

```python
python3 create_midi.py model96256.pth/output.txt
```

## 4. Generate mp3 file

```python
timidity output/output.midi -Ow -o - | ffmpeg -i - -acodec libmp3lame -ab 320k output/output.mp3
timidity output/complete.midi -Ow -o - | ffmpeg -i - -acodec libmp3lame -ab 320k output/complete.mp3
```

The final generated Bach-style music is in the output folder.

# Reference
1. [Generating Original Classical Music with an LSTM Neural Network and Attention](https://medium.com/@alexissa122/generating-original-classical-music-with-an-lstm-neural-network-and-attention-abf03f9ddcb4)
2. [Generating Long-Term Structure in Songs and Stories](https://magenta.tensorflow.org/2016/07/15/lookback-rnn-attention-rnn)
3. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
4. [Baroque Music Generator](https://github.com/Pudkip/Bach-Bot)
5. [Generating Music using LSTM](https://www.researchgate.net/publication/351502178_Generating_Music_using_LSTM)
6. [Music Generation using Deep Learning](https://medium.com/@sabadejuyee21/music-generation-using-deep-learning-7d3dbb2254af)
7. [How to Generate Music using a LSTM Neural Network in Keras](https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5)
8. [Classical-Piano-Composer](https://github.com/Skuldur/Classical-Piano-Composer)
9. [Classic_Music_Generation](https://github.com/thebeyonder001/Classic_Music_Generation)
10. [conversion-tool](https://www.conversion-tool.com/audiotomidi/)

