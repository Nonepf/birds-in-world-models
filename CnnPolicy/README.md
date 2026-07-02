# CNN Policy

This folder implement a simple CnnPolicy AI Player. All the parts have already been implemented by `stable_baselines3`. It is just a test on the environment, and helps to generate training data for future work.

## Usage

run `python train_cnn.py` to train the model, and run `python cnn_demo.py` to see what the model has learned.

When training, run `tensorboard --logdir ./tb_logs/` to view the training loss.