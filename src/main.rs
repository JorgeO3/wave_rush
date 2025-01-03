use std::fs::File;
use wave_rush::*;

// TODO: Check if this is the best way to use the SourceStream
// TODO: take a look at the `lib2.rs` file

fn main() -> Result<()> {
    let file = File::open("input.wav")?;
    let wav_reader = WavReader::try_new(file)?;
    let mut wav_decoder = WavDecoder::try_new(wav_reader)?;

    let time = std::time::Instant::now();
    std::hint::black_box(|| {
        for packet in wav_decoder.packets() {
            let _ = packet.unwrap();
        }
    })();
    println!("Elapsed time: {:?}", time.elapsed());

    Ok(())
}
