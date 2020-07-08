# tweetRT
tweetRT is a extensible API wrapper around twitter and tensorflow, that allows text sentiment
analysis on real time twitter data.

It is a distributed and parallel RNN LSTM that can be used on any localhost or clustered system.
Once the data is trained you can restore the model and use it to view the real time analysis on a twitter feed.
The twitter feeds are filtered by a **value**.. e.i Uber, News, Google, Technology.  

The distributed system is spun up by Process Servers, or PS. Which communicate with the Worker Servers.
Process servers will handle the communications with workers, who will compute all the intensive tasks,
which will spread batches of data across each worker.

### Requirements - Local

* Java 8
* Python

### How to install - Local

```angular2html
> git clone this-repository
> pip install -e /path/to/tweetRT
```

This will install all the dependencies needed to run tweetRT

you should now be able to run
`tweetRT -h`

### How to run

##### GPU on Heracles.

You're going to want three separate sessions inside of **Heracles**
```angular2html
> ssh joshua.brummet@heracles.ucdenver.pvt
> ssh node18
> ll
```

Feel free to scp tweetRT package to your own heracles node18 user account.

First we will make sure that there isn't a `models` directory that exits.
If there is, tensorflow will read the `checkpoint` file and notify the API that it
already a finished graph and there's no training that needs to be done.

I will make sure that there isn't `models` directory before any runs (if using my heracles account).
 If you are scp the package to your own,
there will be fold 'pretrained-models' directory, please remove this or move it into another location on the file system.
But please check before so you can have a successful train.

You should now see the directory `twitter_tensorRT`
please navigate to `/twitter_tensorRT/tweetRT/lstm/`
check to see if the directory `models` exits.
If it does please remove it with a `rm -rf`.
Once there isn't a models directory. You can now run the program.

If you want to test multiple different runs. Just delete the models, to train a new set of data.

Now you should be ready to run on heracles.

**Start a PS**
```angular2html
> tweetRT start server ps --index=0
```

In the second session you can now start a **worker**

```angular2html
> tweetRT start server worker --index=0
```

Finally in the third session you can run.

```angular2html
> tweetRT start server worker --index=1
```

You can now watch each worker train on separate batches of data. It should only be a couple minutes until finished.

You now have trained a data set on a distributed tensorflow model.

###Run sentiment analysis with live twitter feed.

You are unable to use the sentiment analysis on Heracles.

The biggest reason being that it requires a Java JVM version 8.
This is because the program uses spark, which requires this JVM.

To use your own model you will need to run the following command to restore the model.
`python checkpoint_convert.py models/checkpoints/model.ckpt-1002 converted-checkpoints/models-1002.ckpt`

You can now scp `converted-checkpoints` to your local machine to be used by the program. move it into /path/to/tweetRT/lstm/
and rename it as `models`

Otherwise... I have provided a pre-trained model in the program in which you can run from you local device.

Run the following command to start
```angular2html
> tweetRT start stream --filter=<filter> --port=<port>
```
where port and filter are user defined.

example: tweetRT start stream --filter=uber --port=5555

you will see in the output that will describe the next step.
copy and paste the green output in a new tab.

it should be `tweetRT run --port=<port>`

Let the program run and you will see the sentiment output on a live twitter feed.
