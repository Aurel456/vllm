
import subprocess
import torch
from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset

# Check for ffmpeg
try:
    subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL)
except FileNotFoundError:
    print("Error: FFmpeg is not installed. Please install it to use VibeVoice ASR.")
    print("  apt-get install ffmpeg")
    exit(1)

# Model configuration
model_name = "microsoft/VibeVoice-ASR"

# Initialize vLLM
llm = LLM(
    model=model_name,
    trust_remote_code=True,
    # VibeVoice requires bfloat16 or float16
    dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
    limit_mm_per_prompt={"audio": 1},
)

# Download a sample audio file
# We use Mary had a little lamb sample
audio_asset = AudioAsset("mary_had_lamb")
audio_path = audio_asset.path

print(f"Processing audio file: {audio_path}")

# Construct prompt
# VibeVoice uses specific prompting structure
prompt = "Describe the audio:"  # This is a dummy prompt, VibeVoice is mainly ASR

# For ASR, the prompt usually doesn't matter as much as the audio,
# but consistent with VibeVoice usage:
# The model expects <|AUDIO|> token which vLLM handles via input processor
prompt = "<|AUDIO|>\n"

# Define sampling parameters
sampling_params = SamplingParams(
    max_tokens=256,
    temperature=0.0,  # Greedy decoding for ASR
)

# Create inputs
inputs = {
    "prompt": prompt,
    "multi_modal_data": {
        "audio": audio_path
    }
}

# Generate
outputs = llm.generate(inputs, sampling_params=sampling_params)

# Print results
for o in outputs:
    print(f"Generated text: {o.outputs[0].text}")
