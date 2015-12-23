Referencing the TensorFlow seq2seq guide is probably important for background

The checkpoint file exceeds Github's 100MB limit, so I didn't include it. It would be in train\_512\_2.

To train the model:
python chatbot.py --data_dir /data/sls/scratch/juesato/neural_chatbot/data2/ --train_dir ../train_512_2/ --size 512 --num_layers=2 --max_train_data_size 1500000

And if there were a checkpoint file included, you could perform decoding with:
python chatbot.py --data_dir /data/sls/scratch/juesato/neural_chatbot/data2/ --train_dir ../train_512_2/ --size 512 --num_layers=2 --max_train_data_size 1500000 --decode


