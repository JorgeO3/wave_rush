use wave_rush::*;

const INPUT: &str = "input.wav";

fn main() -> Result<()> {
    let file = std::fs::File::open(INPUT)?;
    let wav_reader = WavReader::try_new(file)?;
    let mut wav_decoder = WavDecoder::try_new(wav_reader)?;
    let mut packets = wav_decoder.packets()?;

    while let Some(packet) = packets.next()? {
        let _packet = packet;
    }

    Ok(())
}
