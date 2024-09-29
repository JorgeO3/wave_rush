use std::fs::File;
use wave_rush::*;

fn main() -> Result<()> {
    let file = File::open("dulce_carita.wav")?;
    let source_stream = SourceStream::new(file);
    let wav_reader = WavReader::try_new(source_stream)?;

    println!("Audio format: {:?}", wav_reader);

    // Read the audio data.

    Ok(())
}
