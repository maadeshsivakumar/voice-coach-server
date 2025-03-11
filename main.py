import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Voice Coach API",
              description="API that provides valuable feedback for sales-rep in the audio to improve sales",
              version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
load_dotenv()


class Metrics(BaseModel):
    talk_to_listen_ratio: str
    call_duration: str
    sentiment_rep: str
    sentiment_customer: str
    # script_adherence: str


class AnalyticsResponse(BaseModel):
    summary: str
    metrics: Metrics
    feedback: str
    recommendations: list[str]


gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key)

PROMPT = ("You are an AI Sales Coaching Assistant that processes recorded audio from in-person sales sessions. Your "
          "task is to transcribe, analyze, and generate actionable insights from the audio file. Use your analysis to "
          "help sales reps improve their performance through detailed metrics and personalized feedback. Do not "
          "respond with anything outside of the sales audio. The metrics like duration and talk_to_listen_ratio, "
          "should be calculated exactly, and should be returned as same any number of times the audio is analyzed. "
          "Do not make stuff up. If the given audio does not sound like a sale data, mention that in feedback. Use "
          "N/A when applicable. Steps: 1."
          "**Transcription & Segmentation:**  - Transcribe the entire audio into text.  - Segment the conversation "
          "into key phases (e.g., introduction, needs assessment, objection handling, closing). 2. **Metrics "
          "Extraction:**  - Calculate the talk-to-listen ratio (i.e., percentage of time the rep speaks versus the "
          "customer).  - Identify adherence to the provided sales script or best practices.  - Perform sentiment "
          "analysis to gauge both the rep’s and customer's tone.  - Extract other relevant metrics (e.g., "
          "call duration, number of questions asked, energy level, clear value proposition statements). 3. **Summary "
          "Generation:**  - Create a concise summary that highlights the key points and flow of the conversation.  - "
          "Note any standout moments or potential missed opportunities. 4. **Custom Feedback & Coaching:**  - Based "
          "on the analysis, provide customized feedback addressing strengths and areas for improvement.  - Include "
          "actionable recommendations (e.g., “Try to listen more actively when the customer expresses concerns” or "
          "“Consider emphasizing financing options during the closing phase”).  - Your feedback should be "
          "constructive, supportive, and directly linked to the conversation metrics and content. 5. **Output "
          "Format:**  - Return your results in a structured JSON format with the following keys:  - \"summary\": A "
          "brief textual summary of the call.  - \"metrics\": An object containing all extracted metrics (e.g., "
          "talk-to-listen ratio, sentiment scores, duration, etc.).  - \"feedback\": Custom coaching feedback "
          "tailored to the rep’s performance.  - \"recommendations\": Actionable next steps to improve future calls. "
          "Constraints:  - Ensure your feedback is data-driven and specific to the content of the audio.  - Provide "
          "clear and supportive advice, keeping in mind that the goal is to help improve sales performance.  - Do not "
          "mention any limitations of the audio or analysis process; focus on actionable insights. Example Output: { "
          "\"summary\": \"The call began with a warm greeting and moved into identifying the customer's needs. The "
          "rep presented the product, handled objections briefly, and closed by discussing financing options.\", "
          "\"metrics\": { \"talk_to_listen_ratio\": \"60:40\", \"call_duration\": \"8 minutes 20 seconds\", "
          "\"sentiment_rep\":"
          "\"Positive\", \"sentiment_customer\": \"Neutral\", \"script_adherence\": \"85%\" }, \"feedback\": \"The "
          "rep was energetic and persuasive, but could benefit from asking more open-ended questions to better "
          "uncover customer needs. There was a slight rush in the closing phase.\", \"recommendations\": [ \"Increase "
          "customer listening time to improve engagement.\", \"Slow down the closing segment to ensure all customer "
          "objections are addressed.\", \"Integrate more open-ended questions during the needs assessment phase.\" ] }")

ALLOWED_AUDIO_TYPES = ["audio/wav", "audio/mp3", "audio/aiff", "audio/aac", "audio/ogg", "audio/flac", "audio/mpeg"]


@app.post("/analyze", response_model=AnalyticsResponse)
async def analyze_audio(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(status_code=400, detail="Invalid audio file type")
    audio_data = await file.read()
    audio_part = types.Part.from_bytes(data=audio_data, mime_type=file.content_type)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[PROMPT, audio_part],
        config={
            "response_mime_type": "application/json",
            "response_schema": AnalyticsResponse.schema()
        }
    )
    return response.parsed


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
