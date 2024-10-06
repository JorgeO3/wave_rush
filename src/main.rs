use rayon::prelude::*;
use std::fs::File;
use wave_rush::*;

fn main() -> Result<()> {
    let file = File::open("dulce_carita.wav")?;
    let source_stream = SourceStream::new(file);
    let mut wav_reader = WavReader::try_new(source_stream)?;

    let data_size = (wav_reader.data_end() - wav_reader.data_start()) as usize;

    let time = std::time::Instant::now();
    let data: Vec<_> = wav_reader
        .packets()
        .par_bridge() // Usa rayon para paralelizar el iterador.
        .flatten()
        .collect();

    println!("Tiempo de procesamiento: {:?}", time.elapsed());
    println!("Data size: {}", data.len());
    Ok(())
}
