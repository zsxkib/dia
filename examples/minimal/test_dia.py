import soundfile as sf
from dia.model import Dia

# Create model
model = Dia.from_pretrained("nari-labs/Dia-1.6B")

# Generate dialogue
text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices."
output = model.generate(text)

# Save output
sf.write("output.mp3", output, 44100)
print("Generated audio saved to output.mp3")