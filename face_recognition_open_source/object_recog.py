import os
from groundingdino.util.inference import load_model, load_image, predict, annotate
import numpy as np
from PIL import Image
import groundingdino.datasets.transforms as T
import cv2
import matplotlib.pyplot as plt

HOME = os.getcwd()
CONFIG_PATH = "C:/Users/ETRI/PycharmProjects/pythonProject1/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
#print(CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))

WEIGHTS_PATH = "C:/Users/ETRI/PycharmProjects/pythonProject1/GroundingDINO/groundingdino/weights/groundingdino_swint_ogc.pth"
#print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))
model = load_model(CONFIG_PATH, WEIGHTS_PATH)

def inference(img, prompt, box_threshold=0.65, text_threshold=0.55): #defalut: 0.66, 0.56
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_transformed, _ = transform(Image.fromarray(img), None)

    boxes, logits, phrases = predict(
        model=model,
        image=image_transformed,
        caption=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    annotated_frame = annotate(image_source=img, boxes=boxes, logits=logits, phrases=phrases)
    print(logits)
    print(phrases) # 여기서 safety glasses나 lab coat만 따로 처리할 수 있을 거 같은데...


    return annotated_frame

#TEXT_PROMPT = "A person who wearing safety glasses and a laboratory coat"
#TEXT_PROMPT = "A person who wearing safety glasses and white lab coat with buttons"
TEXT_PROMPT = "A person with safety glasses,a lab coat and mask"

img = cv2.imread("C:/Users/ETRI/PycharmProjects/pythonProject1/GroundingDINO/labortory.jpg")
#C:/Users/ETRI/PycharmProjects/pythonProject1/GroundingDINO/labortory.jpg
#C:/Users/ETRI/PycharmProjects/pythonProject1/GroundingDINO/lab_court.jpg
#C:/Users/ETRI/PycharmProjects/pythonProject1/GroundingDINO/nothing.jpg
#C:/Users/ETRI/PycharmProjects/pythonProject1/GroundingDINO/only_lab_coat.jpg
#C:/Users/ETRI/PycharmProjects/pythonProject1/GroundingDINO/only_safety_glasses.jpg

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result_img = inference(img, TEXT_PROMPT)

plt.figure(figsize=(16, 16))
plt.imshow(result_img[:, :, ::-1])
plt.axis("off")
plt.show()