from typing import Union
from dotenv import load_dotenv
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pdf2image
import pytesseract
import re
import json
import os
from openai import OpenAI
from pytesseract import Output
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

IMAGEDIR = "scanned/"

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
)

def pdf_to_img(pdf_file):
    return pdf2image.convert_from_path(pdf_file)

def plot_gray(image):
    plt.figure(figsize=(16, 10))
    return plt.imshow(image, cmap="Greys_r")


def plot_rgb(image):
    plt.figure(figsize=(16, 10))
    return plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def output_to_json(parsed_data, filename):
    with open(filename, "w") as json_file:
        json.dump(parsed_data, json_file, indent=4)

@app.post("/receipt-data")
def receipt_parse(file: UploadFile = File(...)):
    contents = file.file.read()

    
    with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
        f.write(contents)
        

    file_name = f"{IMAGEDIR}/{file.filename}"
    
    image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    plot_gray(image)

    pytesseract.pytesseract.tesseract_cmd = (
        r"tessaract/tesseract.exe"
    )
    d = pytesseract.image_to_data(image, output_type=Output.DICT)
    n_boxes = len(d["level"])
    boxes = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    for i in range(n_boxes):
        (x, y, w, h) = (d["left"][i], d["top"][i], d["width"][i], d["height"][i])
        boxes = cv2.rectangle(boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plot_rgb(boxes)

    extracted_text = pytesseract.image_to_string(image)
    # print(extracted_text)

    def parseNaGPT(extracted_text):
        receipt_data = {
            "name_of_issuer": "",
            "mobile_number": "",
            "tin": "",
            "vrn": "",
            "serial_no": "",
            "uin": "",
            "tax_office": "",
            "customer": {
                "customer_name": "",
                "customer_id_type": "",
                "customer_id": "",
                "customer_vrn": "",
                "customer_mobile": "",
            },
            "tax": {
              "total_exclusive_of_tax": "",
              "tax_type": "if TAX A put it like this (TAX A - 18%) & if TAX B put it like this (TAX B- 0%)",
              "tax_amount if can not be seen just subtract of total_inclusive_of_tax - total_exlusive_of_tax": "",
              "total_tax if can not be seen just subtract of total_inclusive_of_tax - total_exlusive_of_tax": "",
              "total_inclusive_of_tax": "",
            },
            "receipt_no": "",
            "z_number": "",
            "date": "",
            "time": "",
            "cash": "",
            "items_number": "",
            "receipt_items": [{
                "description":"",
                "price": "as an integer",
                "qty":"",
                "type": "",
            }],
            "receipt_verification_code": "",
        }
        contentGPT = "I want to parse the following text {} to be parse to json with the following format: {}".format(extracted_text, receipt_data)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": contentGPT,
                }
            ],
            model="gpt-3.5-turbo",
        )
        
        return chat_completion.choices[0].message.content
    
    # print(parseNaGPT(extracted_text))

    return json.loads(parseNaGPT(extracted_text))
