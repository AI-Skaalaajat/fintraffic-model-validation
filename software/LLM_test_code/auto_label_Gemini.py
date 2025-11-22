from PIL import Image
from google import genai
from google.genai import types
import glob

# img = Image.open("picture/C01507_2025-10-04T00-51-18+00-00.jpg")

client = genai.Client(api_key="")
paths = glob.glob("picture/*.jpg")
imgs = [Image.open(p) for p in paths]
for img in imgs:
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=["Based on driving safety standards, determine whether the road surface is dry (0) or wet (1)", img],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),  # Disables thinking
            system_instruction=(
                "output only 0(dry) and 1(wet). "
                "If the judgment cannot be made, output -1,"
                "no other content is allowed."
            ),
            max_output_tokens=2,
            temperature=0.0
        ),
    )
    if response.text == '-1':
        print(img.filename, ":", "unclear")
    elif response.text == '0':
        print(img.filename, ":", "dry")
    elif response.text == '1':
        print(img.filename, ":", "wet")
