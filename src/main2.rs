use std::fs::File;
use wave_rush::*;

fn main() -> Result<()> {
    let file = File::open("dulce_carita.wav")?;
    let mut wav_reader = WavReader::open(file)?;

    let time = std::time::Instant::now();
    std::hint::black_box(|| for _packet in wav_reader.packets() {})();
    println!("Elapsed: {:?}", time.elapsed());

    Ok(())
}
