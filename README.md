End-To-End Memory Networks in Tensorflow applied to the Ubuntu Dialog Corpus
============================================================================

This is a Tensorflow implementation of the [End-to-End Memory Network](http://arxiv.org/abs/1503.08895v4) applied to the [Ubuntu Dialog Corpus](https://arxiv.org/abs/1506.08909). The model can be compared to a LSTM and a [DSSM](https://www.semanticscholar.org/paper/Learning-deep-structured-semantic-models-for-web-Huang-He/5b9534442f91a87022427b74bca9fd95dd045383), both available within the implementation.


Data
----

Data must be [downloaded](https://drive.google.com/open?id=0ByBCKvHbumEFbXVUUjctOVVwakk) independently and decompressed in the data folder.

Once you get the data, you need to create the TFRecords that will be used to speed up training:

    $ python3 data/prepare_data.py


Running the code
----------------

To train the Memory Network, simply run the following command:

    $ python3 train.py
    
In case you want to use the other implementations, you can run:

    $ python3 train.py --network=LSTM
    $ python3 train.py --network=DSSM
    
Paramters can be tuned in config.py. The default configuration is also detailed in the file.

To test the network after a training, you need to run:

    $ python3 test.py
    
Once again, you can specify the type of network to test, in case you did not train on the Memory Network. It is also possible to the directory of the models to load, such as:

    $ python3 test.py --network=DSSM --model_dir=path/to/my/tensorflow/checkpoints
    
Finally, it is possible to test the network on a single randomly chosen context, by running:

    $ python3 predict.py


Evaluation
----------

We achieve the following performance:

| Network           | r@1          | r@2          | r@5          |
| ----------------- | ------------ | ------------ | ------------ |
| MN 				      	| 49,01%       | 65,49%       | 87,77%       |
| DSSM 			      	| 38,16%       | 55,55%       | 83,41%       |
| LSTM 			      	| 58,30%       | 74,56%       | 93,11%       |


Author
------

Rémi Vincent / [@remeus](https://github.com/Remeus)

