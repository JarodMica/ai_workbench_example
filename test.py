from pathlib import Path
from f5_tts_local.model import DiT, UNetT
from f5_tts_local.infer.utils_infer import load_vocoder, load_model
from f5_tts_local.infer.infer_cli import main_process

vocos = load_vocoder(is_local=False)

model = "F5-TTS"
model_cls = DiT
model_config = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
checkpoint_file = "model_1200000.pt"
vocab_file = "f5_tts_local/infer/examples/vocab.txt"

ema_model = load_model(model_cls, model_config, checkpoint_file, vocab_file)

ref_audio = "f5_tts_local/infer/examples/basic/basic_ref_en.wav"
ref_text = "Some call me nature, others call me mother nature."
gen_text = "I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring."
remove_silence=False
speed=1.0
output_dir = Path("results")
wave_path = Path(output_dir) / "test.wav"

main_process(ref_audio, ref_text, gen_text, ema_model, remove_silence, speed, output_dir, wave_path)