use std::fs::File;
use wave_rushlib::*;

// TODO: Check if this is the best way to use the SourceStream
// TODO: take a look at the `lib2.rs` file

fn main() -> Result<()> {
    // let file = File::open("dulce_carita.wav")?;
    // let file = File::open("big_input.wav")?;
    // let path = std::env::args().nth(1).expect("No file path provided");
    // let file = File::open(path)?;
    let file = File::open("long_input.wav")?;
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
