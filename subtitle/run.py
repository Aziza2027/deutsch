import os
import shutil
import subprocess
import configparser
from faster_whisper import WhisperModel
import datetime
import srt
import shutil
import logging
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate


from tqdm import tqdm
from deep_translator import GoogleTranslator
from subprocess import call

import torchaudio

def read_config(config_file):
    """
    Reads the configuration file and returns the input and output folder paths, model size,
    device, compute type, and generate_transcript option as a tuple.
    """
    config = configparser.ConfigParser()
    config.read(config_file)

    input_folder = config.get("PATHS", "input_folder")
    output_folder = config.get("PATHS", "output_folder")
    model_size = config.get("MODEL", "model_size")
    device = config.get("MODEL", "device")
    compute_type = config.get("MODEL", "compute_type")
    task = config.get("MODEL", "task")
    generate_transcript = config.getboolean("OPTIONS", "generate_transcript")
    generate_summary = config.getboolean("OPTIONS", "generate_summary")
    summary_language = config.get("OPTIONS", "summary_language")
    openai_api_key = config.get("OPENAI", "API_KEY")
    openai_api_base = config.get("OPENAI", "API_BASE")
    openai_model = config.get("OPENAI", "model")
    text_chunk_size = config.getint("OPTIONS", "text_chunk_size")
    max_chunk = config.getint("OPTIONS", "max_chunk")

    return (
        input_folder,
        output_folder,
        model_size,
        device,
        compute_type,
        task,
        generate_transcript,
        generate_summary,
        summary_language,
        openai_api_key,
        openai_api_base,
        openai_model,
        text_chunk_size,
        max_chunk
    )


def convert_to_mp3(video_file, audio_file):
    """
    Converts the input video file to MP3 format using ffmpeg and saves the resulting audio file to
    the specified output file path. Returns True if the conversion is successful, False otherwise.
    """
    if os.path.exists(audio_file):
        print(f"Skipping {video_file} - audio file already exists")
        return True
    
    # if file is mp3, just use shutil to copy it
    if video_file.endswith(".mp3"):
        print(f"Copying {video_file} to {audio_file}")
        shutil.copy2(video_file, audio_file)
        return True

    cmd = [
        "ffmpeg",
        "-i",
        video_file,
        "-vn",
        "-acodec",
        "libmp3lame",
        "-ab",
        "192k",
        "-ac",
        "2",
        "-loglevel",
        "quiet",
        audio_file,
    ]

    print(f"Processing: {video_file}")
    print(f"Output audio path: {audio_file}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"Conversion of {video_file} failed")
        return False

    print(f"Conversion of {video_file} success")
    return True

def generate_subtitles(model, task, audio_file, subtitle_file, transcript_file=None, subtitle_file2=None, translate_to=None):
    """
    Transcribes the specified audio file using the Faster-Whisper model, generates an SRT subtitle file
    at the specified output file path, and optionally generates a transcript file. Returns True if the
    subtitle generation is successful, False otherwise.
    """
    # if os.path.exists(subtitle_file):
    #     st.write(f"Skipping {audio_file} - subtitles file already exists")
    #     return True

    # Transcribe the audio file using the Faster-Whisper model
    segments, info = model.transcribe(audio_file, vad_filter=True, task=task)

    st.write(
        f"Detected language '{info.language}' with probability {info.language_probability}"
    )

    subtitles = []
    texts = []

    subtitles2 = []
    waveform, sample_rate = torchaudio.load(audio_file)

    # Get the length in seconds
    audio_length = waveform.size(1) / sample_rate

    progress_text = "Operation in progress. Please wait."

    my_bar = st.progress(0, text=progress_text)

    # Generate subtitles for each segment
    for i, segment in enumerate(segments):
        
        start_time = datetime.timedelta(milliseconds=segment.start * 1000)
        end_time = datetime.timedelta(milliseconds=segment.end * 1000)
        text = segment.text.strip()
        texts.append(text)

        my_bar.progress(
            segment.end/audio_length if segment.end/audio_length < 1 else 1.0, 
            text=progress_text)

        if text:
            # Create subtitle objects for both original and translated text
            subtitle = srt.Subtitle(
                index=i + 1, start=start_time, end=end_time, content=text
            )
            subtitles.append(subtitle)

            if translate_to:
                text_en = GoogleTranslator(source=info.language, target=translate_to).translate(text)

                subtitle2 = srt.Subtitle(
                    index=i + 1, start=start_time, end=end_time, content=text_en
                )
                subtitles2.append(subtitle2)

    my_bar.progress(1.0, text=progress_text)


    # Write the subtitles to the SRT file
    with open(subtitle_file, "w", encoding="utf-8") as f:
        f.write(srt.compose(subtitles))
    
    with open(subtitle_file2, "w", encoding="utf-8") as f:
        f.write(srt.compose(subtitles2))

    st.write(f"Generation of {subtitle_file} successful")

    # Write the transcript file if requested
    if transcript_file:
        with open(transcript_file, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text + " ")
        st.write(f"Generation of {transcript_file} successful")

    return True


def summarize(
    transcript_file,
    summary_file,
    summary_language,
    openai_api_key,
    openai_api_base,
    openai_model,
    text_chunk_size,
    max_chunk
):
    # Instantiate the LLM modelI
    if os.path.exists(summary_file):
        print(f"Skipping {summary_file} - summary file already exists")
        return True
    print("Strat summaring " + transcript_file)

    llm = ChatOpenAI(
        model=openai_model,
        temperature=0.7,
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base,
    )
    with open(transcript_file, "r") as file:
        txt = file.read()
    # print(txt)
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=text_chunk_size,
    )
    texts = text_splitter.split_text(txt)
    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]
    # Text summarization
    # Write a concise summary of the following
    # CONCISE SUMMARY IN {language}:"""

    if len(docs) == 1:
        stuff_prompt_template = """Please use Markdown syntax to help me summarize the key information and important content. Your response should summarize the main information and important content in the original text in a clear manner, using appropriate headings, markers, and formats to facilitate readability and understanding.Please note that your response should retain the relevant details in the original text while presenting them in a concise and clear manner. You can freely choose the content to highlight and use appropriate Markdown markers to emphasize it. Now summary following content in {language}:

        {text}

        """
        stuff_prompt = PromptTemplate(
            template=stuff_prompt_template,
            input_variables=[
                "text",
            ],
            partial_variables={"language": summary_language},
        )

        chain = load_summarize_chain(
            llm,
            chain_type="stuff",
            prompt=stuff_prompt
        )
        response = chain.run(docs)
        print(response)
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(response)
        return True

    if(len(docs) > max_chunk):
        print('The doc is too long, you should use gpt4-4k or calude to summarize it')
        return True
    map_prompt_template = """Write a concise summary of the following:


    {text}


    SUMMARY IN {language}:"""

    map_prompt = PromptTemplate(
        template=map_prompt_template,
        input_variables=[
            "text",
        ],
        partial_variables={"language": summary_language},
    )

    combine_prompt_template = """Write a concise summary of the following:


    {text}


    CONCISE SUMMARY IN {language}:"""

    combine_prompt = PromptTemplate(
        template=combine_prompt_template,
        input_variables=[
            "text",
        ],
        partial_variables={"language": summary_language},
    )

    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        return_intermediate_steps=True,
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
    )

    response = chain({"input_documents": docs}, return_only_outputs=True)
    print(response)
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Summary_text:\n")
        f.write(response["output_text"])
        f.write("\n\nSection Contents:\n\n")
        for idx, section in enumerate(response["intermediate_steps"]):
            f.write(f"{idx + 1}.{section}\n")
    return True






import streamlit as st
import os
import logging
from faster_whisper import WhisperModel

# Read the configuration from the config file
config_file = "example.config.ini"
(
    input_folder,
    output_folder,
    model_size,
    device,
    compute_type,
    task,
    generate_transcript,
    generate_summary,
    summary_language,
    openai_api_key,
    openai_api_base,
    openai_model,
    text_chunk_size,
    max_chunk
) = read_config(config_file)

# Initialize the Streamlit app
st.title("Video Subtitle Generator")

# Display the configuration parameters, allowing the user to edit them
st.subheader("Modify Configuration Parameters (if needed):")

#  Upload file for processing
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "wmv", "flv", "mkv", "mp3"])


model_sizes = [
    "tiny.en", "tiny", "base.en", "base", "small.en", 
    "small", "medium.en", "medium", "large-v1", "large-v2"
]

# Dropdown to select the model size with 'small' as the default
translate_to = st.selectbox("Choose the language to translate", ['ru','en','de','uz', None], index=4)


model_size = st.selectbox("Model Size", model_sizes, index=model_sizes.index("tiny"))
# translate_to = st.selectbox("Choose the language to translate", ['ru','en','de','uz', None], index=4)

# Asking the user if they want a delay
want_delay = st.radio("Do you want a delay?", ("Yes", "No"), index=1)

# If the user wants a delay, ask for the number of seconds
if want_delay == "Yes":
    delay_seconds = st.number_input("How many seconds of delay?", min_value=0, step=1)



# text_chunk_size = st.number_input("Text Chunk Size", value=text_chunk_size)
# max_chunk = st.number_input("Max Chunk", value=max_chunk)

#
if uploaded_file is not None:

    save_path = f"./data/input/{uploaded_file.name}"

    # Save the uploaded file to the file system
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    audio_extensions = ['.mp3', '.wav', '.ogg', '.aac', '.flac', '.m4a']

    # Get the file extension
    file_extension = '.' + save_path.split('.')[-1].lower()
    file_name = uploaded_file.name

    if file_extension in audio_extensions:
        st.write("Audio file. Converting to video...")
        command_aud = f'ffmpeg -y -f lavfi -i color=color=black:size=1280x720:rate=25 -i "{save_path}" -c:v libx264 -c:a aac -strict experimental -b:a 192k -shortest "{save_path}.mp4"'
        
        save_path = f'{save_path}.mp4'
        file_name = save_path.split('/')[-1]
    
    video_file = file_name

    audio_file = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".mp3")
    subtitle_file = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".srt")
    subtitle_file2 = os.path.join(output_folder, os.path.splitext(file_name)[0] + "2.srt")
    transcript_file = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".txt") if generate_transcript else None
    summary_file = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".md") if generate_summary else None

    if st.button("Process"):
        if file_extension in audio_extensions:
            call(command_aud, shell=True)
        # Set up logging and model
        logging.basicConfig()
        logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
        model = WhisperModel(model_size, device=device)

        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Process video
        if convert_to_mp3(save_path, audio_file):
            st.success("Audio extraction successful!")
            st.text("Converting video to MP3...")
            generate_subtitles(model, task, audio_file, subtitle_file, transcript_file, subtitle_file2, translate_to)

            st.success("Subtitles generated!")
            st.text("Preparing your video...")

            # Generate the final video with subtitles
            command = f'ffmpeg -y -i "{save_path}" -vf "subtitles=\'{subtitle_file}\':force_style=\'Alignment=2\',subtitles=\'{subtitle_file2}\':force_style=\'Alignment=6\'" -c:a copy "{save_path}.mp4"'
            if translate_to is None:
                # Command without the second subtitle
                command = f'ffmpeg -y -i "{save_path}" -vf "subtitles=\'{subtitle_file}\':force_style=\'Alignment=2\'" -c:a copy "{save_path}.mp4"'
            
            if want_delay:
                delay_command = f'ffmpeg -y -itsoffset {delay_seconds} -i {subtitle_file} -c copy {subtitle_file}.srt'
                os.system(delay_command)
                shutil.move(f'{subtitle_file}.srt', subtitle_file)

                if translate_to != None:
                    delay_command = f'ffmpeg -y -itsoffset {delay_seconds} -i {subtitle_file2} -c copy {subtitle_file2}.srt'
                    os.system(delay_command)
                    shutil.move(f'{subtitle_file2}.srt', subtitle_file2)

            print(command)
            os.system(command)

            st.success("Video with subtitles generated!")
            output_video = f"{save_path}.mp4"
            # Display download link
            with open(output_video, "rb") as video:
                st.download_button("Download Final Video", video, file_name=video_file)
