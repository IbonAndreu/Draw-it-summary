import os
from quickdraw import QuickDrawDataGroup
import tensorflow as tf
import splitfolders


def create_output_folders():
    """
    Create output folders for model, metrics, and params, if they don't exist
    """

    dir = f'{os.environ.get("LOCAL_REGISTRY_PATH")}/'
    if not os.path.exists(dir):
            os.mkdir(dir)

    dir = f'{os.environ.get("LOCAL_REGISTRY_PATH")}/metrics/'
    if not os.path.exists(dir):
            os.mkdir(dir)

    dir = f'{os.environ.get("LOCAL_REGISTRY_PATH")}/models/'
    if not os.path.exists(dir):
            os.mkdir(dir)

    dir = f'{os.environ.get("LOCAL_REGISTRY_PATH")}/params/'
    if not os.path.exists(dir):
            os.mkdir(dir)


def create_img_data() -> None:
    """
    Download image data using QuickDrawDataGroup package and store locally
    """

    print('start downloading')
    image_size = (64, 64)
    max_drawings = int(os.environ.get("MAX_DRAWINGS"))

    #Create initial data folder if it does not exist
    dir = f'{os.environ.get("LOCAL_DATA_PATH")}/'
    if not os.path.exists(dir):
            os.mkdir(dir)

    dir = f'{os.environ.get("LOCAL_DATA_PATH")}/initial_data/'
    if not os.path.exists(dir):
            os.mkdir(dir)

    #function to download images
    def create_images(name, max_drawings, recognized):
        images = QuickDrawDataGroup(name, max_drawings=max_drawings, recognized=recognized)
        for img in images.drawings:
            file = f'{os.environ.get("LOCAL_DATA_PATH")}/initial_data/{name}/{str(img.key_id)}.png'
            img.get_image(stroke_width=2).resize(image_size).save(file)

    #List of desired images
    names = ['The Eiffel Tower', 'The Great Wall of China', 'The Mona Lisa', 'aircraft carrier', 'airplane', 'alarm clock', 'ambulance', 'angel', 'animal migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday cake', 'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling fan', 'cell phone', 'cello', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise ship', 'cup', 'diamond', 'dishwasher', 'diving board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger', 'fire hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight', 'flip flops', 'floor lamp', 'flower', 'flying saucer', 'foot', 'fork', 'frog', 'frying pan', 'garden', 'garden hose', 'giraffe', 'goatee', 'golf club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey puck', 'hockey stick', 'horse', 'hospital', 'hot air balloon', 'hot dog', 'hot tub', 'hourglass', 'house', 'house plant', 'hurricane', 'ice cream', 'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop', 'leaf', 'leg', 'light bulb', 'lighter', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail', 'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paint can', 'paintbrush', 'palm tree', 'panda', 'pants', 'paper clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup truck', 'picture frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police car', 'pond', 'pool', 'popsicle', 'postcard', 'potato', 'power outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain', 'rainbow', 'rake', 'remote control', 'rhinoceros', 'rifle', 'river', 'roller coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school bus', 'scissors', 'scorpion', 'screwdriver', 'sea turtle', 'see saw', 'shark', 'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping bag', 'smiley face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer ball', 'sock', 'speedboat', 'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 'stop sign', 'stove', 'strawberry', 'streetlight', 'string bean', 'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing set', 'sword', 'syringe', 't-shirt', 'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis racquet', 'tent', 'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado', 'tractor', 'traffic light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing machine', 'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine bottle', 'wine glass', 'wristwatch', 'yoga', 'zebra', 'zigzag']

    #Checks if folders exist and downloads images
    for name in names:
        dir = f'{os.environ.get("LOCAL_DATA_PATH")}/initial_data/{name}'
        if not os.path.exists(dir):
            os.mkdir(dir)
        create_images(name, max_drawings, recognized=True)


def create_split_folders() -> None:
    """
    Split the initial data into test, train and val folders
    """

    #Get splits for train, val and test (% of data)
    train_share = float(os.environ.get("TRAIN_SHARE"))
    val_share = float(os.environ.get("VAL_SHARE"))
    test_share = float(os.environ.get("TEST_SHARE"))

    #Create split data folder, if it does not exist
    dir = f'{os.environ.get("LOCAL_DATA_PATH")}/split_data'
    if not os.path.exists(dir):
            os.mkdir(dir)

    #Take images from initial_data folder and split into train, val, test
    splitfolders.ratio(f'{os.environ.get("LOCAL_DATA_PATH")}/initial_data', # The location of dataset
                    output=f'{os.environ.get("LOCAL_DATA_PATH")}/split_data', # The output location
                    seed=42, # The number of seed
                    ratio=(train_share, val_share, test_share), # The ratio of splited dataset
                    group_prefix=None, # If your dataset contains more than one file like ".jpg", ".pdf", etc
                    move=False # If you choose to move, turn this into True
                    )


def get_tensor_data():
    """
    Return a tf.data.Dataset based on the locally stored data
    """

    batch_size = int(os.environ.get("DATA_BATCH_SIZE"))

    #Create tf.data.Dataset from local files
    if os.environ.get("DATA_SOURCE") == "local":

       # train_df
        train_ds = tf.keras.utils.image_dataset_from_directory(
            f'{os.environ.get("LOCAL_DATA_PATH")}/split_data/train',
            seed=42,
            label_mode='categorical',
            image_size=(64, 64),
            batch_size=batch_size,
            color_mode='rgb'
        )

       # val_df
        val_ds = tf.keras.utils.image_dataset_from_directory(
            f'{os.environ.get("LOCAL_DATA_PATH")}/split_data/val',
            seed=42,
            label_mode='categorical',
            image_size=(64, 64),
            batch_size=batch_size,
            color_mode='rgb'
        )

        # test_df
        test_ds = tf.keras.utils.image_dataset_from_directory(
            f'{os.environ.get("LOCAL_DATA_PATH")}/split_data/val',
            seed=42,
            label_mode='categorical',
            image_size=(64, 64),
            batch_size=batch_size,
            color_mode='rgb'
        )

        return train_ds, val_ds, test_ds
