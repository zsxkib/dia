# Dia

This is a pip-installable fork of the original [Nari Labs Dia repository](https://github.com/nari-labs/dia).

Dia is a 1.6B parameter text-to-speech model that directly generates highly realistic dialogue from a transcript.

## Installation

```bash
pip install git+https://github.com/cccntu/dia.git
```

## Basic Usage

```python
import soundfile as sf
from dia.model import Dia

# Create model
model = Dia.from_pretrained("nari-labs/Dia-1.6B")

# Generate dialogue
text = "[S1] Dia is a text to speech model. [S2] You can create conversations easily!"
output = model.generate(text)

# Save output
sf.write("output.mp3", output, 44100)
```

## Features

- Generate dialogue using `[S1]` and `[S2]` tags
- Generate non-verbal sounds like `(laughs)`, `(coughs)`, etc.
- Voice cloning by conditioning with audio

## Examples

Check out the [examples](examples/) directory for usage examples:
- [Minimal example](examples/minimal/): A minimal example of using Dia in a separate project

## Requirements

- CUDA-compatible GPU with at least 10GB VRAM
- PyTorch 2.0+ with CUDA 12.6

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.