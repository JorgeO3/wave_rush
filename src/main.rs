use std::fs::File;
use wave_rush::*;

// TODO: Check if this is the best way to use the SourceStream
// TODO: take a look at the `lib2.rs` file

fn main() -> Result<()> {
    let file = File::open("dulce_carita.wav")?;
    let source_stream = SourceStream::new(file);
    let mut wav_reader = WavReader::try_new(source_stream)?;

    let time = std::time::Instant::now();
    std::hint::black_box(|| for _packet in wav_reader.packets() {})();
    println!("Elapsed: {:?}", time.elapsed());

    Ok(())
}
