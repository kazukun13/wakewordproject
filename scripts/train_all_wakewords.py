from openwakeword.train import train_model

# Train "Hey Room" model
train_model(
    positive_folder="data/training_data/finalheyroom",
    negative_folder="data/training_data/finalnotwakeword",
    model_output_path="models/hey_room.tflite",
    wakeword_name="hey_room"
)

# Train "Hi Room" model
train_model(
    positive_folder="data/training_data/finalhiroom",
    negative_folder="data/training_data/finalnotwakeword",
    model_output_path="models/hi_room.tflite",
    wakeword_name="hi_room"
)

# Train "Wake Up Room" model
train_model(
    positive_folder="data/training_data/finalwakeuproom",
    negative_folder="data/training_data/finalnotwakeword",
    model_output_path="models/wakeup_room.tflite",
    wakeword_name="wakeup_room"
)