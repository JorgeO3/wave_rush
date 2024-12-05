use std::fs::File;
use wave_rushv2::*;

// TODO: Check if this is the best way to use the SourceStream
// TODO: take a look at the `lib2.rs` file

fn main() -> Result<()> {
    let file = File::open("dulce_carita.wav")?;
    let wav_reader = WavReader::try_new(file)?;
    let mut wav_decoder = WavDecoder::try_new(wav_reader)?;

    for sample in wav_decoder.samples() {
        dbg!(sample)?;
    }

    Ok(())
}
