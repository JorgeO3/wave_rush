use wave_rush::*;

fn main() -> Result<()> {
    // let file = std::fs::File::open("./data/dulce_carita.wav")?;
    let file = std::fs::File::open("./data/input.wav")?;
    let wav_reader = WavReader::try_new(file)?;
    let mut wav_decoder = WavDecoder::try_new(wav_reader)?;
    let mut packets = wav_decoder.packets();

    while let Some(packet) = packets.next()? {
        let _packet = packet;
    }

    let data: Vec<i32> = Vec::new();

    Ok(())
}
