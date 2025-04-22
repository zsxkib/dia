# Minimal Dia Example

This example demonstrates how to use Dia in a separate project after installation.

## Setup

```bash
# Install dia from pip
pip install git+https://github.com/cccntu/dia.git

# Run the example
python test_dia.py
```

The script will generate an audio file `output.mp3` with the synthesized dialogue.

## Code Walkthrough

```python
import soundfile as sf
from dia.model import Dia

# Create model
model = Dia.from_pretrained("nari-labs/Dia-1.6B")

# Generate dialogue
text = "[S1] Dia is a text to dialogue model. [S2] You can create conversations easily!"
output = model.generate(text)

# Save output
sf.write("output.mp3", output, 44100)
```