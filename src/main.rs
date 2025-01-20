use std::fs::File;
use wave_rush::*;

fn main() -> Result<()> {
    let file = File::open("input.wav")?;
    let wav_reader = WavReader::try_new(file)?;
    let mut wav_decoder = WavDecoder::try_new(wav_reader)?;

    let time = std::time::Instant::now();
    for packet in wav_decoder.packets() {
        let _ = packet.unwrap();
    }
    println!("Elapsed time: {:?}", time.elapsed());

    Ok(())
}
