
#imports
from turtle import width
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import random
import requests
from streamlit_lottie  import st_lottie, st_lottie_spinner
import os

# Page title, layout and page icon
st.set_page_config(page_title="Can you draw it?", page_icon='✏️' ,layout="wide")

# set columns
col1, col2 = st.columns([3, 3])
new_col, new_col1 = st.columns(2)

# Set sessionstates
if 'random_word' not in st.session_state:
    st.session_state['random_word'] = ''
if 'running' not in st.session_state:
    st.session_state['running'] = 0
if 'submit' not in st.session_state:
    st.session_state['submit'] = 0
if 'score' not in st.session_state:
    st.session_state['score'] = 0
if 'canvas_key' not in st.session_state:
    st.session_state['canvas_key'] = 0
if 'result' not in st.session_state:
    st.session_state['result'] = ''

#List of Symbols
list = ['The Eiffel Tower', 'The Great Wall of China', 'The Mona Lisa', 'aircraft carrier', 'airplane', 'alarm clock', 'ambulance', 'angel', 'animal migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday cake', 'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling fan', 'cell phone', 'cello', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise ship', 'cup', 'diamond', 'dishwasher', 'diving board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger', 'fire hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight', 'flip flops', 'floor lamp', 'flower', 'flying saucer', 'foot', 'fork', 'frog', 'frying pan', 'garden', 'garden hose', 'giraffe', 'goatee', 'golf club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey puck', 'hockey stick', 'horse', 'hospital', 'hot air balloon', 'hot dog', 'hot tub', 'hourglass', 'house', 'house plant', 'hurricane', 'ice cream', 'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop', 'leaf', 'leg', 'light bulb', 'lighter', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail', 'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paint can', 'paintbrush', 'palm tree', 'panda', 'pants', 'paper clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup truck', 'picture frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police car', 'pond', 'pool', 'popsicle', 'postcard', 'potato', 'power outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain', 'rainbow', 'rake', 'remote control', 'rhinoceros', 'rifle', 'river', 'roller coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school bus', 'scissors', 'scorpion', 'screwdriver', 'sea turtle', 'see saw', 'shark', 'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping bag', 'smiley face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer ball', 'sock', 'speedboat', 'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 'stop sign', 'stove', 'strawberry', 'streetlight', 'string bean', 'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing set', 'sword', 'syringe', 't-shirt', 'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis racquet', 'tent', 'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado', 'tractor', 'traffic light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing machine', 'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine bottle', 'wine glass', 'wristwatch', 'yoga', 'zebra', 'zigzag']

# Front page, logo, robot saying 'hello' and 'Play!'- Button
with new_col:
    m = st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #FFFFFF;color:black; font-weight: bold;font-size:20px;height:3em;width:15em;border-radius:30px 30px 30px 30px;
    }
    </style>""", unsafe_allow_html=True)

with new_col1:
    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    with col1:
        if st.session_state['running'] ==0:
            image = Image.open('logo.png')
            st.image(image, width=600)

    with col2:
        if st.session_state['running'] ==0:
            lottie_url_hello = "https://assets5.lottiefiles.com/packages/lf20_3vbOcw.json"
            lottie_hello = load_lottieurl(lottie_url_hello)
            st_lottie(lottie_hello, key="hello", height=450, width=450)
        if st.session_state['running'] ==0:
            if col2.button("Play!", key="id1"):
                st.session_state['random_word'] = random.choice(list)
                st.session_state['running'] = 1
                st.session_state['canvas_key'] = st.session_state['canvas_key']+1 #change to reset canvas

# Build up the main app
if st.session_state['running'] ==1:
    random_word = st.session_state['random_word']
    vowel = 'aeiou'

    # Can you draw it? header
    col1.markdown("<h1 style='text-align: left; color: #388A73;'>Can you draw it?</h1>", unsafe_allow_html=True)

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}style="font-size:300%;</style>', unsafe_allow_html=True)

    # box 'easy', 'medium', 'hard'
    difficulty = col1.radio("", ('Easy', 'Medium', 'Hard'))

    # 'Time to draw: {random word}'
    if random_word[0].lower() in vowel:
        col1.subheader(f'Time to draw an: ')
        col1.markdown(f'<h1 style="text-align:left;font-family:Tahoma;color:#3d85c6;font-size:30px;">{random_word}</h1>', unsafe_allow_html=True)
    else:
        col1.subheader(f'Time to draw a: ')
        col1.markdown(f'<h1 style="text-align:left;font-family:Tahoma;color:#3d85c6;font-size:30px;">{random_word}</h1>', unsafe_allow_html=True)

    #Create canvas
    with col2:
        col2.write(' ')
        col2.write(' ')
        col2.write(' ')
        col2.write(' ')
        col2.write(' ')

        canvas_result = st_canvas(
        fill_color="#eee",
        stroke_width=3,
        drawing_mode='freedraw',
        stroke_color='black',
        background_color='#FFFFFF',
        update_streamlit=True,
        height=350,
        width=350,
        display_toolbar=True,
        key=st.session_state['canvas_key']
    )
        #Submit Button
        if st.button("Submit", key="id2"):
            st.session_state['submit'] = 1
            img_data = canvas_result.image_data
            im = Image.fromarray(img_data.astype('uint8'), mode="RGBA")
            im.save("test.png", "PNG")
            filename = "test.png"
            files = {'file': (filename, open(filename, 'rb'))}
            response = requests.post(
                os.environ.get("API_URL"),
                files=files,
            )
            x=response.json()
            st.session_state['result'] = x[0:10]

            # logic behind 'easy', 'medium', 'hard'
            if difficulty == 'Hard':
            # check string starts with vowel or consonant
                if x[0].lower() in vowel and x == st.session_state['random_word'] in x[0:1]:
                    st.subheader(f'I recognised it! It is an {random_word}')
                    st.balloons()
                    col2.button("Next item", key="id7")

                elif x[0].lower() not in vowel and st.session_state['random_word'] in x[0:1]:
                    st.subheader(f'I recognised it! It is a {random_word}')
                    st.balloons()
                    st.session_state['score'] = st.session_state['score']+1

                else:
                    st.subheader("Oh no! I couldn't figure it out.")

            if difficulty == 'Medium':
            # check string starts with vowel or consonant
                if x[0].lower() in vowel and st.session_state['random_word'] in x[0:5]:
                    st.subheader(f'I recognised it! It is an {random_word}')
                    st.balloons()
                    st.session_state['score'] = st.session_state['score']+1

                elif x[0].lower() not in vowel and st.session_state['random_word'] in x[0:5]:
                    st.subheader(f'I recognised it! It is a {random_word}')
                    st.balloons()
                    st.session_state['score'] = st.session_state['score']+1

                else:
                    st.subheader("Oh no! I couldn't figure it out.")

            if difficulty == 'Easy':
            # check string starts with vowel or consonant
                random_word = st.session_state['random_word']

                if random_word[0].lower() in vowel and random_word in x[0:10]:
                    st.subheader(f'I recognised it! It is an {random_word}')
                    st.balloons()
                    st.session_state['score'] = st.session_state['score']+1

                elif random_word[0].lower() not in vowel and random_word in x[0:10]:
                    st.subheader(f'I recognised it! It is a {random_word}')
                    st.balloons()
                    st.session_state['score'] = st.session_state['score']+1

                else:
                    st.subheader("Oh no! I couldn't figure it out.")

            # 'Next item'- Button
        if col2.button("Next item"):
            st.session_state['random_word'] = random.choice(list)
            st.session_state['canvas_key'] = st.session_state['canvas_key']+1 #change to reset canvas
            st.experimental_rerun()

        # 'show similar results of last submit'
        if col2.checkbox("Show similar results of last submit"):
            if st.session_state['result']=='':
                pass
            else: st.write(st.session_state['result'][0],', ',st.session_state['result'][1],', ',st.session_state['result'][2],', ',st.session_state['result'][3],', ',st.session_state['result'][4],', ',st.session_state['result'][5],', ',st.session_state['result'][6],', ',st.session_state['result'][7],', ',st.session_state['result'][8],', ',st.session_state['result'][9])

        # 'your current score' - shows animation of the number
        with col1:
            if st.session_state['score'] ==1:

                score = st.session_state['score']

            score = st.session_state['score']

            if score ==1:
                st.write("." * 90)
                col1.subheader(f'Your current score: ')
                lottie_url_hello = "https://assets1.lottiefiles.com/packages/lf20_Az1u8i.json"
                lottie_hello = load_lottieurl(lottie_url_hello)
                st_lottie(lottie_hello, key="hello",height=150,
                width=150,)
            if score ==2:
                st.write("." * 90)
                col1.subheader(f'Your current score: ')
                lottie_url_hello = "https://assets8.lottiefiles.com/packages/lf20_2MnYNd.json"
                lottie_hello = load_lottieurl(lottie_url_hello)
                st_lottie(lottie_hello, key="hello",height=150,
                width=150,)
            if score ==3:
                st.write("." * 90)
                col1.subheader(f'Your current score: ')
                lottie_url_hello = "https://assets8.lottiefiles.com/packages/lf20_DCXHIu.json"
                lottie_hello = load_lottieurl(lottie_url_hello)
                st_lottie(lottie_hello, key="hello",height=150,
                width=150,)
            if score ==4:
                st.write("." * 90)
                col1.subheader(f'Your current score: ')
                lottie_url_hello = "https://assets8.lottiefiles.com/packages/lf20_9STzCM.json"
                lottie_hello = load_lottieurl(lottie_url_hello)
                st_lottie(lottie_hello, key="hello",height=150,
                width=150,)
            if score ==5:
                st.write("." * 90)
                col1.subheader(f'Your current score: ')
                lottie_url_hello = "https://assets8.lottiefiles.com/packages/lf20_QxUOmv.json"
                lottie_hello = load_lottieurl(lottie_url_hello)
                st_lottie(lottie_hello, key="hello",height=150,
                width=150,)
            if score ==6:
                st.write("." * 90)
                col1.subheader(f'Your current score: ')
                lottie_url_hello = "https://assets8.lottiefiles.com/packages/lf20_6Jp14J.json"
                lottie_hello = load_lottieurl(lottie_url_hello)
                st_lottie(lottie_hello, key="hello",height=150,
                width=150,)
            if score ==7:
                st.write("." * 90)
                col1.subheader(f'Your current score: ')
                lottie_url_hello = "https://assets8.lottiefiles.com/packages/lf20_CuYFG1.json"
                lottie_hello = load_lottieurl(lottie_url_hello)
                st_lottie(lottie_hello, key="hello",height=150,
                width=150,)
            if score ==8:
                st.write("." * 90)
                col1.subheader(f'Your current score: ')
                lottie_url_hello = "https://assets8.lottiefiles.com/packages/lf20_r9YWxm.json"
                lottie_hello = load_lottieurl(lottie_url_hello)
                st_lottie(lottie_hello, key="hello",height=150,
                width=150,)
            if score ==9:
                st.write("." * 90)
                col1.subheader(f'Your current score: ')
                lottie_url_hello = "https://assets8.lottiefiles.com/packages/lf20_fHel3A.json"
                lottie_hello = load_lottieurl(lottie_url_hello)
                st_lottie(lottie_hello, key="hello",height=150,
                width=150,)
            if score ==10:
                st.write("." * 90)
                col1.subheader(f'You won!!!')
                lottie_url_hello = "https://assets5.lottiefiles.com/packages/lf20_joexwr4o.json"
                lottie_hello = load_lottieurl(lottie_url_hello)
                st_lottie(lottie_hello, key="hello",height=150,
                width=300,)

                # 'start again' - Button after player won
                if col1.button('Start again!', key="12"):
                    st.session_state['random_word'] = random.choice(list)
                    st.session_state['running'] = 1
                    st.session_state['canvas_key'] = st.session_state['canvas_key']+1

                    st.session_state['score'] = 0
                    score = st.session_state['score']
