# Activity-recognition-on-MHEALTH

Activity recognition using MHEALTH dataset and CNN or FNN

# Whats inside

  - Data preprocessing files
  - Dataset in TXT 
  - CNN model
  - FNN model
  - other tools

### Tech insturments

All frameworks, libraries and tools that I have used:

* [PyTorch](https://pytorch.org)
* [Tensorflow](https://www.tensorflow.org) - *using Keras API*
* [Sendgrid](https://www.tensorflow.org) - email notfication when model training is done
* [Tensorboard](https://github.com/tensorflow/tensorboard) - data visualization in Tensorflow

### Files description

Dillinger is currently extended with the following plugins. Instructions on how to use them in your own application are linked below.

| File | Description |
| ------ | ------ |
| data_preprocessing.py | using for preprocess data, creating WINDOWS, deleting "0" labels, normalization, shuffling data |
| data_preprocessing(7-features).py | same as previous, but added Сompression X Y Z axis into one spatial vector |
| data/ | Folder for dataset files and models |
| data/main_CNN.py | CNN model file |
| data/main_NNLL.py  | NNL model file |
| data/logs | logs dir for data visualization in Tensorboard (only for CNN) |
| sendgrid.py | Email notification about model training ending  |

### How to start Tensorboard
Tensorboard is instrument for data visualization in real-time. It is using callbacks and logs dir.

To start tensorboard its needed to have folder with logs files

```sh
cd data
tensorboard --logdir=logs
```

Result will be something like this

```sh
dyn172-30-203-79:data kinivi$ tensorboard --logdir=logs
W0809 12:59:49.608335 123145369452544 plugin_event_accumulator.py:294] Found more than one graph event per run, or there was a metagraph containing a graph_def, as well as one or more graph events.  Overwriting the graph with the newest event.
TensorBoard 1.14.0 at http://MacBook-Pro-Nikita-2.local:6006/ (Press CTRL+C to quit)
```

This will start instance and now you can open in in browser. Adress `localhost:6060` or other port that in results of previous command

Verify the starting

```sh
127.0.0.1:6060
```


License
----

MIT


**Author: Nikita Kiselov**


