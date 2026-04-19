import os
import time
from dotenv import load_dotenv
import requests
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
import azure.cognitiveservices.speech as speechsdk
from msrest.authentication import CognitiveServicesCredentials
from flask import Flask, render_template, request

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

# Read Azure Computer Vision credentials
VISION_KEY = os.getenv("VISION_KEY")
VISION_ENDPOINT = os.getenv("VISION_ENDPOINT")

if not VISION_KEY or not VISION_ENDPOINT:
    raise ValueError("Missing VISION_KEY or VISION_ENDPOINT in .env file.")

# Create Computer Vision client
vision_client = ComputerVisionClient(
    VISION_ENDPOINT,
    CognitiveServicesCredentials(VISION_KEY)
)

#reading the speech credentials
SPEECH_KEY = os.getenv("SPEECH_KEY")
SPEECH_REGION = os.getenv("SPEECH_REGION")

if not SPEECH_KEY or not SPEECH_REGION:
    raise ValueError("Missing SPEECH_KEY or SPEECH_REGION in .env file.")

#-----------------------------------------------------


def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from a local image using Azure Computer Vision OCR.
    Returns all detected text as a single string.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(image_path, "rb") as image_stream:
        read_response = vision_client.read_in_stream(image_stream, raw=True)

    operation_location = read_response.headers["Operation-Location"]
    operation_id = operation_location.split("/")[-1]

    # Wait for OCR result
    while True:
        read_result = vision_client.get_read_result(operation_id)
        if read_result.status not in ["notStarted", "running"]:
            break
        time.sleep(1)

    if read_result.status != "succeeded":
        return ""

    extracted_lines = []
    for page in read_result.analyze_result.read_results:
        for line in page.lines:
            extracted_lines.append(line.text)

    return "\n".join(extracted_lines)


# if __name__ == "__main__":
#     test_image = "test_image.jpg"  # change if needed
#     text = extract_text_from_image(test_image)
#     print("Extracted OCR Text:")
#     print(text)

#--------------------------------------------------------------------------
import requests

# Read Azure Language credentials
LANGUAGE_KEY = os.getenv("LANGUAGE_KEY")
LANGUAGE_ENDPOINT = os.getenv("LANGUAGE_ENDPOINT")

if not LANGUAGE_KEY or not LANGUAGE_ENDPOINT:
    raise ValueError("Missing LANGUAGE_KEY or LANGUAGE_ENDPOINT in .env file.")

# CLU configuration
CLU_PROJECT_NAME = "PrescriptionAssistantCLU"
CLU_DEPLOYMENT_NAME = "deployment1"
CLU_PREDICTION_URL = (
    f"{LANGUAGE_ENDPOINT}/language/:analyze-conversations?api-version=2023-04-01"
)


def analyze_user_query_with_clu(user_text: str) -> dict:
    """
    Send user text to the deployed CLU model and return
    the top intent, confidence, all intents, and extracted entities.
    """
    headers = {
        "Ocp-Apim-Subscription-Key": LANGUAGE_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "kind": "Conversation",
        "analysisInput": {
            "conversationItem": {
                "id": "1",
                "text": user_text,
                "modality": "text",
                "language": "en",
                "participantId": "1"
            }
        },
        "parameters": {
            "projectName": CLU_PROJECT_NAME,
            "deploymentName": CLU_DEPLOYMENT_NAME,
            "verbose": True,
            "stringIndexType": "TextElement_V8"
        }
    }

    response = requests.post(CLU_PREDICTION_URL, headers=headers, json=payload)
    response.raise_for_status()

    result = response.json()
    prediction = result["result"]["prediction"]

    return {
        "top_intent": prediction["topIntent"],
        "intents": prediction.get("intents", []),
        "entities": prediction.get("entities", [])
    }

#-----------------------------------------------------------------------------------
def process_prescription_request(image_path: str = None, user_text: str = None) -> dict:
    if not image_path and not user_text:
        return {
            "success": False,
            "message": "Please provide either an image path or a user text query."
        }

    ocr_text = ""
    clu_result = None

    if image_path:
        ocr_text = extract_text_from_image(image_path)

    if user_text:
        clu_result = analyze_user_query_with_clu(user_text)

    top_intent = None
    entities = []

    if clu_result:
        top_intent = clu_result.get("top_intent")
        entities = clu_result.get("entities", [])

    response_message = generate_demo_response(
        top_intent=top_intent,
        ocr_text=ocr_text,
        entities=entities
    )

    return {
        "success": True,
        "ocr_text": ocr_text,
        "clu_result": clu_result,
        "response_message": response_message
    }

#-----------------------------------------------------------
def generate_demo_response(top_intent: str, ocr_text: str = "", entities: list = None) -> str:
    """
    Generate a simple user-facing response based on CLU intent,
    OCR text, and extracted entities.
    """
    entities = entities or []
    lower_ocr = ocr_text.lower() if ocr_text else ""

    disclaimer = (
        "\n\nDisclaimer: This is a demo assistant only and is not a substitute "
        "for advice from a doctor or pharmacist."
    )

    if top_intent == "ExplainPrescription":
        if ocr_text:
            return (
                "Here is a simplified explanation of the prescription:\n\n"
                f"{ocr_text}"
                f"{disclaimer}"
            )
        return (
            "Please upload a prescription image so I can extract and explain the text."
            f"{disclaimer}"
        )

    elif top_intent == "AskDosage":
        if ocr_text:
            medicine_lines = []
            for line in ocr_text.split("\n"):
                if any(keyword in line.lower() for keyword in ["tab", "cap", "syr", "inj"]):
                    medicine_lines.append(line)

            if medicine_lines:
                joined_lines = "\n".join(f"- {line}" for line in medicine_lines)
                return (
                    "I found these medicine instruction lines in the prescription:\n\n"
                    f"{joined_lines}\n\n"
                    "These lines may indicate dosage or timing instructions, "
                    "but please confirm the exact dosage with your doctor or pharmacist."
                    f"{disclaimer}"
                )

        return (
            "I can help interpret dosage instructions if you upload a prescription image "
            "or provide the medicine instruction text."
            f"{disclaimer}"
        )

    elif top_intent == "AskMedicinePurpose":
        medicine_names = [e["text"] for e in entities if e.get("category") == "MedicineName"]

        if medicine_names:
            med_list = ", ".join(medicine_names)
            return (
                f"You appear to be asking about the purpose of: {med_list}. "
                "This demo assistant can help identify medicine-related text from a prescription, "
                "but you should confirm the medicine’s purpose with a pharmacist or doctor."
                f"{disclaimer}"
            )

        if ocr_text:
            return (
                "I can see medicine names or prescription lines in the uploaded text, "
                "but this demo version does not verify clinical medicine purpose from a trusted drug database."
                f"{disclaimer}"
            )

        return (
            "Please upload a prescription image or mention the medicine name you want help with."
            f"{disclaimer}"
        )

    elif top_intent == "AskUsageClarification":
        timing_entities = [e["text"] for e in entities if e.get("category") == "TimingInstruction"]

        if timing_entities:
            phrase = timing_entities[0].lower()

            if "after food" in phrase:
                return (
                    "“After food” usually means the medicine should be taken after eating a meal."
                    f"{disclaimer}"
                )
            elif "before food" in phrase or "before meals" in phrase:
                return (
                    "“Before food” usually means the medicine should be taken before eating."
                    f"{disclaimer}"
                )
            elif "at night" in phrase:
                return (
                    "“At night” usually means the medicine should be taken in the evening or before sleep."
                    f"{disclaimer}"
                )

        return (
            "I can help explain prescription phrases like timing or usage instructions in simpler language."
            f"{disclaimer}"
        )

    elif top_intent == "GeneralHelp":
        return (
            "You can upload a prescription image and ask me to explain it, clarify dosage instructions, "
            "or simplify phrases such as 'after food'."
            f"{disclaimer}"
        )

    return (
        "I understood your request, but I could not generate a detailed response yet."
        f"{disclaimer}"
    )

#---------------------------------------------------------------------------------
#transcribing function (speech to text)
def transcribe_audio_file(audio_path: str) -> str:
    """
    Transcribe a local WAV audio file using Azure Speech-to-Text.
    Returns recognized text if successful, otherwise an empty string.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION
    )
    speech_config.speech_recognition_language = "en-US"

    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    result = recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text

    return ""
    




#---------------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    image = request.files.get("image")
    user_text = request.form.get("user_text")

    image_path = None

    if image and image.filename != "":
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], image.filename)
        image.save(image_path)

    result = process_prescription_request(
        image_path=image_path,
        user_text=user_text
    )

@app.route("/process_voice", methods=["POST"])
def process_voice():
    audio = request.files.get("audio")
    image = request.files.get("image")
    user_text = request.form.get("user_text")

    audio_path = None
    image_path = None
    recognized_text = ""

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    try:
        if audio and audio.filename != "":
            audio_path = os.path.join(app.config["UPLOAD_FOLDER"], audio.filename)
            audio.save(audio_path)
            recognized_text = transcribe_audio_file(audio_path)

        if image and image.filename != "":
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], image.filename)
            image.save(image_path)

        final_user_text = recognized_text if recognized_text else user_text

        result = process_prescription_request(
            image_path=image_path,
            user_text=final_user_text
        )

        result["recognized_text"] = recognized_text

    except Exception as e:
        result = {
            "success": False,
            "response_message": f"Voice processing failed. {str(e)}",
            "recognized_text": "",
            "ocr_text": ""
        }

    return render_template("index.html", result=result)

#----------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)