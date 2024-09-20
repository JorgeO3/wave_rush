use std::fs::File;
use wave_rush::*;

fn main() -> Result<()> {
    let file = File::open("dulce_carita.wav")?;
    let source_stream = SourceStream::new(file);
    let mut wav_reader = WavReader::try_new(source_stream)?;

    // Read the audio data.
    let audio_data = wav_reader.read_audio_data()?;

    Ok(())
}
