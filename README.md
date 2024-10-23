# How to use

1. Install general requirements with `pip install -r requirements.txt`
2. Either install tensorflow or pytorch with `pip install tensorflow` (`pip install tensorflow[and-cuda]` if you want to use GPU) or `pip install pytorch`
3. Go to [Twitch](https://dev.twitch.tv/console), create an application and add the credentials to `.env`
4. Run `python ./scripts/fetch_messages.py --username <twitch_username>` to fetch chats from all current twitch VODs
5. Run `python ./train_pt.py` or `python ./train_tf.py` to train the model depending on whether you have pytorch or tensorflow installed

Relevant environment variables for training:

- `LOAD_MODEL`: Load saved model instead of training a new one
- `SAVE_MODEL`: Save model after training