import time
from flask import Flask, request, render_template, make_response, session
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from pytube import YouTube, extract
import textwrap
import openai
import os
import logging
import google.generativeai as genai
from dotenv import load_dotenv
from google.cloud import speech
import io

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Add secret key for sessions

transcript = ""
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={
        "temperature": 0.5,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    },
)

# Set your OpenAI API key as an environment variable for security
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key


def get_video_metadata(url):
    """
    Fetch video metadata using pytube.
    Handle missing metadata gracefully.
    """
    try:
        yt = YouTube(url)
        title = yt.title if hasattr(yt, 'title') else "Title unavailable"
        owner = yt.author if hasattr(yt, 'author') else "Owner unavailable"
        description = yt.description if hasattr(yt, 'description') else "Description unavailable"
        return {
            "title": title,
            "owner": owner,
            "description": description,
        }
    except Exception as e:
        logger.error(f"Error fetching video metadata: {e}")
        return {
            "title": "Title unavailable",
            "owner": "Owner unavailable",
            "description": "Description unavailable",
        }



def get_video_id(url):
    try:
        return extract.video_id(url)
    except Exception as e:
        logger.error(f"Error extracting video ID: {e}")
        return None

def download_audio(video_url, output_path="audio.mp3"):
    yt = YouTube(video_url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_stream.download(filename=output_path)
    return output_path


def transcribe_audio(file_path):
    client = speech.SpeechClient()

    with io.open(file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    transcription = []
    for result in response.results:
        transcription.append(result.alternatives[0].transcript)

    return transcription

def fetch_transcript(video_id, video_url):
    """
    Fetch the transcript for a video.
    If English transcription is not available, fetch the auto-generated transcription and translate it into English.
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # First try to get any English transcript (manual or auto-generated)
        try:
            # Try getting any English transcript
            transcript = transcript_list.find_transcript(['en'])
            return transcript.fetch()
        except NoTranscriptFound:
            # If no English transcript, try getting any available transcript
            try:
                available_transcripts = transcript_list.find_generated_transcript(['en', 'ja', 'ko', 'es', 'fr', 'de', 'hi'])
                if available_transcripts:
                    non_english_transcript = available_transcripts.fetch()
                    logger.info("Found non-English transcript. Attempting translation.")
                    return translate_transcript_to_english(non_english_transcript)
            except Exception as e:
                logger.error(f"Error fetching available transcripts: {str(e)}")
                return None
                
    except TranscriptsDisabled:
        logger.error("Transcripts are disabled for this video.")
        try:
            output_path = download_audio(video_url)
            return transcribe_audio(output_path)
        except Exception as e:
            logger.info(f"Audio Error: {str(e)}")
            return None
    except NoTranscriptFound:
        logger.error("No transcripts found for this video.")
        return None
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        return None


def translate_transcript_to_english(transcript):
    """
    Translate a transcript to English using an AI model.
    """
    try:
        # Combine transcript text into one string
        full_text = ' '.join([entry['text'] for entry in transcript])
        
        # Use AI model to translate
        prompt = f"""
        Translate the following transcript into English:
        {full_text}
        """
        
        response = gemini.generate_content(prompt)
        translated_text = response.text.strip()
        
        # Return translated transcript in the original format
        return [{"text": line} for line in translated_text.split('\n') if line.strip()]
        
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return [{"text": "Translation failed. Original transcript: " + ' '.join([entry['text'] for entry in transcript])}]


def translate_text(text):
    """
    Translate text to English using the AI model.
    """
    try:
        prompt = f"""
        Translate the following text into English:
        {text}
        """
        response = gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return text  # Return original text if translation fails


def summarize_text(text):
    """
    Summarize text using the AI model.
    """
    try:
        system_instructions = """You are a skilled content creator and assistant specializing in summarizing long-form content and transforming it into engaging, reader-friendly blogs. Your task is to analyze the provided YouTube video transcript, extract key points, and craft a well-structured blog post that captivates readers.
            Guidelines for the Blog:

            Summary: Begin with a concise introduction summarizing the video's main topic and purpose. Highlight why it’s relevant or valuable to the reader.
            Key Points: Break down the transcript into logical sections, extracting essential ideas, insights, and takeaways. Use bullet points or subheadings for clarity where appropriate.
            Tone: Maintain an informative, engaging, and conversational tone throughout the blog. Adjust the tone based on the video's subject (e.g., professional for a tech tutorial, lighthearted for lifestyle content).
            Actionable Insights: Provide actionable tips or advice wherever applicable, helping readers apply the video’s content to their own lives or work.
            Conclusion: End with a strong conclusion summarizing the key messages and encouraging readers to engage further (e.g., watch the video, comment, share).
            SEO Optimization: Suggest a compelling title and meta description for the blog to optimize it for search engines.
            Deliverable Format:

            Provide the blog in markdown format with appropriate headings, subheadings, and formatting.
            Include a list of keywords for SEO and suggest a title that grabs attention.
            Transcript:"""
        text = system_instructions + "\n" + text
        total_tokens = gemini.count_tokens(text).total_tokens
        logger.info(f"Gemini Token count: {total_tokens}")
        response = gemini.generate_content(text)
        logger.info(f"Response: {response.text}")
        return response.text
    except Exception as e:
        logger.info(f"Exception in Summarize text: {str(e)}")
        return "An error occurred during summarization."


# def generate_summary(transcript):
#     """
#     Generate a summary from the transcript using AI.
#     """
#     full_text = ' '.join([entry['text'] for entry in transcript])
#     chunks = split_text(full_text)
#     summaries = [summarize_text(chunk) for chunk in chunks]
#     return ' '.join(summaries)


@app.route('/', methods=['GET', 'POST'])
def index():
    summary = ""
    transcript_text = ""
    if request.method == 'POST':
        global transcript
        logger.info("Loading...")
        video_url = request.form['url']
        video_id = get_video_id(video_url)
        logger.info(f"Attempting to fetch transcript for video ID: {video_id}")
        
        if video_id:
            transcript = fetch_transcript(video_id, video_url)
            if transcript:
                transcript_text = ' '.join([entry['text'] for entry in transcript])
            else:
                logger.error(f"Failed to fetch transcript for video ID: {video_id}")
                summary = "No transcripts available for this video. Please try another video."
                return render_template('index.html', summary=summary)

            summary = summarize_text(transcript_text)
            logger.info(f"Transcript summarized successfully {summary}")
            return render_template('index.html', summary=summary)
        else:
            summary = "Invalid YouTube URL."

    return render_template('index.html', summary=summary)


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get('question', '').strip()

        if not question:
            return {"response": "Please provide a valid question."}, 400

        # Use the transcript from the global variable
        global transcript
        context = transcript
        logger.info(f"Transcript : {len(context)}")
        if not context:
            return {"response": "Transcript not loaded. Please summarize a video first."}, 400

        # Prepare the prompt
        prompt = f"""
        You are a highly intelligent assistant with knowledge of the following video details:
        {context}
        
        The user has asked: "{question}". Please provide a clear and concise response.
        """

        # Generate a response using the AI model
        response = gemini.generate_content(prompt).text

        return {"response": response}, 200

    except Exception as e:
        logger.exception(f"Error during chat: {str(e)}")
        return {"response": "An error occurred while processing your request."}, 500


if __name__ == '__main__':
    # Ensure the OpenAI API key is set
    if not openai.api_key:
        logger.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        exit(1)

    # Run the Flask app
    app.run(debug=True)
