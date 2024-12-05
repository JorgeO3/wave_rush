use std::collections::HashMap;
use std::fmt::Debug;
use std::io::{self, BufReader, Read, Seek, SeekFrom};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Static(&'static str),
    #[error("Error genérico: {0}")]
    Generic(String),
    #[error(transparent)]
    Io(#[from] io::Error),
}

/// Tipo especializado Result para operaciones en este crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Representa diferentes formatos de muestra.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    Uint8,
    Int16,
    Int24,
    Int32,
}

impl SampleFormat {
    /// Retorna el número de bytes por muestra.
    pub fn bytes_per_sample(&self) -> u16 {
        match self {
            SampleFormat::Uint8 => 1,
            SampleFormat::Int16 => 2,
            SampleFormat::Int24 => 3,
            SampleFormat::Int32 => 4,
        }
    }
}

/// Representa los parámetros del códec para el archivo WAV.
#[derive(Debug, Default)]
struct CodecParams {
    pub sample_rate: Option<u32>,
    pub num_frames: Option<u64>,
    pub sample_format: Option<SampleFormat>,
    pub bits_per_sample: Option<u16>,
    pub num_channels: u16,
    pub block_align: Option<u16>,
    pub audio_format: Option<u16>,
}

/// Una Tag encapsula un par clave-valor de metadatos.
#[derive(Clone, Debug)]
pub struct Tag {
    pub tag_type: Option<TagType>,
    pub key: String,
    pub value: String,
}

/// Tipo de etiqueta basado en claves.
#[derive(Debug, Clone)]
pub enum TagType {
    Rating,
    Comment,
    OriginalDate,
    Genre,
    Artist,
    Copyright,
    Date,
    EncodedBy,
    Engineer,
    TrackTotal,
    Language,
    Composer,
    TrackTitle,
    Album,
    Producer,
    TrackNumber,
    Encoder,
    MediaFormat,
    Writer,
    Label,
    Version,
    Unknown,
}

impl TagType {
    /// Convierte un arreglo de 4 bytes a TagType.
    fn from_bytes(key: &[u8; 4]) -> Self {
        match key {
            b"ages" => TagType::Rating,
            b"cmnt" | b"comm" | b"icmt" => TagType::Comment,
            b"dtim" | b"idit" => TagType::OriginalDate,
            b"genr" | b"ignr" | b"isgn" => TagType::Genre,
            b"iart" => TagType::Artist,
            b"icop" => TagType::Copyright,
            b"icrd" | b"year" => TagType::Date,
            b"ienc" | b"itch" => TagType::EncodedBy,
            b"ieng" => TagType::Engineer,
            b"ifrm" => TagType::TrackTotal,
            b"ilng" | b"lang" => TagType::Language,
            b"imus" => TagType::Composer,
            b"inam" | b"titl" => TagType::TrackTitle,
            b"iprd" => TagType::Album,
            b"ipro" => TagType::Producer,
            b"iprt" | b"trck" | b"prt1" | b"prt2" => TagType::TrackNumber,
            b"isft" => TagType::Encoder,
            b"isrf" => TagType::MediaFormat,
            b"iwri" => TagType::Writer,
            b"torg" => TagType::Label,
            b"tver" => TagType::Version,
            _ => TagType::Unknown,
        }
    }
}

/// Representa el chunk "fmt " en un archivo WAV.
pub struct FormatChunk {
    audio_format: u16,
    num_channels: u16,
    sample_rate: u32,
    byte_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
}

/// Representa el chunk "LIST" en un archivo WAV.
pub struct ListChunk {
    list_type: [u8; 4],
    tags: Vec<Tag>,
}

/// Representa el chunk "fact" en un archivo WAV.
pub struct FactChunk {
    num_samples: u32,
}

/// Representa el chunk "data" en un archivo WAV.
pub struct DataChunk {
    length: u32,
    data_position: u64,
}

/// Enum representando varios tipos de chunks WAV.
pub enum WaveChunk {
    Format(FormatChunk),
    List(ListChunk),
    Fact(FactChunk),
    Data(DataChunk),
}

/// Enum para identificar tipos de chunks basados en su ID.
pub enum ChunkType {
    Format,
    List,
    Fact,
    Data,
    Unknown,
}

impl ChunkType {
    /// Convierte un ID de 4 bytes en el correspondiente ChunkType.
    pub fn from_id(id: &[u8; 4]) -> Self {
        match id {
            b"fmt " => ChunkType::Format,
            b"LIST" => ChunkType::List,
            b"fact" => ChunkType::Fact,
            b"data" => ChunkType::Data,
            _ => ChunkType::Unknown,
        }
    }
}

/// Analiza chunks de un archivo WAV.
struct ChunkParser<'a, R: Read + Seek + Debug> {
    reader: &'a mut BufReader<R>,
    cursor: usize,
    length: usize,
}

impl<'a, R: Read + Seek + Debug> ChunkParser<'a, R> {
    /// Crea un nuevo analizador de chunks con el flujo de origen y longitud dados.
    pub fn new(reader: &'a mut BufReader<R>, length: usize) -> Self {
        Self {
            reader,
            cursor: 0,
            length,
        }
    }

    /// Alinea el cursor al siguiente límite de 2 bytes, si es necesario.
    fn align(&mut self) -> Result<()> {
        if self.cursor & 1 != 0 {
            self.skip_bytes(1)?;
        }
        Ok(())
    }

    /// Mueve el cursor por el número especificado de bytes.
    fn skip_bytes(&mut self, n: usize) -> Result<()> {
        self.reader.seek(SeekFrom::Current(n as i64))?;
        self.cursor += n;
        Ok(())
    }

    #[inline]
    fn read_u16_le(&mut self) -> Result<u16> {
        let bytes = self.read_exact::<2>()?;
        Ok(u16::from_le_bytes(bytes))
    }

    #[inline]
    fn read_u32_le(&mut self) -> Result<u32> {
        let bytes = self.read_exact::<4>()?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_exact<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut result = [0u8; N];
        self.reader.read_exact(&mut result)?;
        self.cursor += N;
        Ok(result)
    }

    fn read_bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut buffer = vec![0; n];
        self.reader.read_exact(&mut buffer)?;
        self.cursor += n;
        Ok(buffer)
    }

    /// Itera sobre cada chunk y aplica la función dada a cada uno.
    pub fn parse_chunks<F>(&mut self, mut f: F) -> Result<()>
    where
        F: FnMut(WaveChunk) -> Result<()>,
    {
        while self.cursor + 8 <= self.length {
            // Lee el ID del chunk y su tamaño.
            let chunk_id = self.read_exact::<4>()?;
            let chunk_size = self.read_u32_le()? as usize;

            // Verifica si el tamaño del chunk excede los bytes restantes.
            if self.length - self.cursor < chunk_size {
                return Err(Error::Static("Chunk size exceeds the remaining length"));
            }

            // Procesa el chunk basado en su ID.
            let chunk = match ChunkType::from_id(&chunk_id) {
                ChunkType::Format => self.parse_format_chunk(chunk_size)?,
                ChunkType::Data => self.parse_data_chunk(chunk_size)?,
                ChunkType::List => self.parse_list_chunk(chunk_size)?,
                ChunkType::Fact => self.parse_fact_chunk(chunk_size)?,
                ChunkType::Unknown => {
                    self.skip_bytes(chunk_size)?;
                    continue;
                }
            };

            f(chunk)?;
            self.align()?;
        }

        Ok(())
    }

    /// Analiza un chunk "fmt ".
    fn parse_format_chunk(&mut self, chunk_size: usize) -> Result<WaveChunk> {
        if chunk_size < 16 {
            return Err(Error::Static("Invalid format chunk size"));
        }

        let audio_format = self.read_u16_le()?;
        let num_channels = self.read_u16_le()?;
        let sample_rate = self.read_u32_le()?;
        let byte_rate = self.read_u32_le()?;
        let block_align = self.read_u16_le()?;
        let bits_per_sample = self.read_u16_le()?;
        let remaining = chunk_size - 16;

        if remaining > 0 {
            self.skip_bytes(remaining)?;
        }

        Ok(WaveChunk::Format(FormatChunk {
            audio_format,
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
        }))
    }

    /// Analiza un chunk "data".
    fn parse_data_chunk(&mut self, chunk_size: usize) -> Result<WaveChunk> {
        let data_position = self.reader.stream_position()?;
        self.skip_bytes(chunk_size)?;
        self.cursor += chunk_size;
        Ok(WaveChunk::Data(DataChunk {
            length: chunk_size as u32,
            data_position,
        }))
    }

    /// Analiza un chunk "LIST".
    fn parse_list_chunk(&mut self, chunk_size: usize) -> Result<WaveChunk> {
        let list_type = self.read_exact::<4>()?;
        let remaining_size = chunk_size - 4;
        let mut tags = Vec::new();

        if &list_type == b"INFO" {
            self.parse_info_chunk(remaining_size, &mut tags)?;
        } else {
            self.skip_bytes(remaining_size)?;
        }

        Ok(WaveChunk::List(ListChunk { list_type, tags }))
    }

    /// Analiza el chunk "INFO", que es un subchunk del chunk "LIST".
    fn parse_info_chunk(&mut self, mut remaining_size: usize, tags: &mut Vec<Tag>) -> Result<()> {
        while remaining_size >= 8 {
            // Lee la clave y el tamaño de la etiqueta.
            let tag_key = self.read_exact::<4>()?;
            let tag_size = self.read_u32_le()? as usize;
            remaining_size -= 8;

            // Lee el valor de la etiqueta.
            let tag_value_bytes = self.read_bytes(tag_size)?;
            let tag_value = String::from_utf8_lossy(&tag_value_bytes)
                .trim_end_matches(char::from(0))
                .to_string();
            remaining_size -= tag_size;

            // Alinea a byte par si es necesario.
            if tag_size % 2 != 0 {
                self.skip_bytes(1)?;
                remaining_size -= 1;
            }

            // Agrega la etiqueta al vector.
            tags.push(Tag {
                tag_type: Some(TagType::from_bytes(&tag_key)),
                key: String::from_utf8_lossy(&tag_key).into_owned(),
                value: tag_value,
            });
        }
        Ok(())
    }

    /// Analiza un chunk "fact".
    fn parse_fact_chunk(&mut self, chunk_size: usize) -> Result<WaveChunk> {
        let num_samples = self.read_u32_le()?;
        if chunk_size > 4 {
            self.skip_bytes(chunk_size - 4)?;
        }
        Ok(WaveChunk::Fact(FactChunk { num_samples }))
    }
}

#[derive(Debug)]
struct WavReaderOptions {
    codec_params: CodecParams,
    metadata: HashMap<String, String>,
    data_start: u64,
    data_end: u64,
}

/// Lector de archivos WAV.
#[derive(Debug)]
pub struct WavReader<R: Read + Seek + Debug> {
    reader: BufReader<R>,
    opts: WavReaderOptions,
}

impl<R: Read + Seek + Debug> WavReader<R> {
    const RIFF_HEADER: [u8; 4] = *b"RIFF";
    const WAVE_HEADER: [u8; 4] = *b"WAVE";
    const BUFFER_SIZE: usize = 1024 * 16;

    /// Intenta crear un nuevo WavReader analizando las cabeceras del archivo WAV.
    pub fn try_new(file: R) -> Result<Self> {
        let mut reader = BufReader::with_capacity(Self::BUFFER_SIZE, file);

        let riff_header = {
            let mut riff_header = [0u8; 4];
            reader.read_exact(&mut riff_header)?;
            riff_header
        };
        if riff_header != Self::RIFF_HEADER {
            return Err(Error::Static("Invalid RIFF header"));
        }

        let chunk_size = {
            let mut size_bytes = [0u8; 4];
            reader.read_exact(&mut size_bytes)?;
            u32::from_le_bytes(size_bytes) as usize
        };

        let wave_header = {
            let mut wave_header = [0u8; 4];
            reader.read_exact(&mut wave_header)?;
            wave_header
        };
        if wave_header != Self::WAVE_HEADER {
            return Err(Error::Static("Invalid WAVE header"));
        }

        let mut options = WavReaderOptions {
            codec_params: CodecParams::default(),
            metadata: HashMap::new(),
            data_start: 0,
            data_end: 0,
        };

        let mut parser = ChunkParser::new(&mut reader, chunk_size);

        parser.parse_chunks(|chunk| match chunk {
            WaveChunk::Format(format) => Self::handle_format_chunk(&mut options, format),
            WaveChunk::Data(data) => Self::handle_data_chunk(&mut options, data),
            WaveChunk::Fact(fact) => Self::handle_fact_chunk(&mut options, fact),
            WaveChunk::List(list) => Self::handle_list_chunk(&mut options, list),
        })?;

        Ok(Self {
            reader,
            opts: options,
        })
    }

    fn handle_format_chunk(options: &mut WavReaderOptions, chunk: FormatChunk) -> Result<()> {
        options.codec_params.sample_rate = Some(chunk.sample_rate);
        options.codec_params.num_channels = chunk.num_channels;
        options.codec_params.bits_per_sample = Some(chunk.bits_per_sample);
        options.codec_params.block_align = Some(chunk.block_align);
        options.codec_params.audio_format = Some(chunk.audio_format);

        let sample_format = match chunk.bits_per_sample {
            8 => SampleFormat::Uint8,
            16 => SampleFormat::Int16,
            24 => SampleFormat::Int24,
            32 => SampleFormat::Int32,
            _ => return Err(Error::Static("Sample format not supported")),
        };
        options.codec_params.sample_format = Some(sample_format);

        Ok(())
    }

    fn handle_fact_chunk(options: &mut WavReaderOptions, chunk: FactChunk) -> Result<()> {
        options.codec_params.num_frames = Some(chunk.num_samples as u64);
        Ok(())
    }

    fn handle_data_chunk(options: &mut WavReaderOptions, chunk: DataChunk) -> Result<()> {
        options.data_start = chunk.data_position;
        options.data_end = chunk.data_position + chunk.length as u64;

        if let Some(block_align) = options.codec_params.block_align {
            let num_frames = chunk.length as u64 / block_align as u64;
            options.codec_params.num_frames = Some(num_frames);
        }
        Ok(())
    }

    fn handle_list_chunk(options: &mut WavReaderOptions, chunk: ListChunk) -> Result<()> {
        for tag in chunk.tags {
            options.metadata.insert(tag.key, tag.value);
        }
        Ok(())
    }

    pub fn num_frames(&self) -> u64 {
        self.opts.codec_params.num_frames.unwrap_or(0)
    }

    pub fn sample_rate(&self) -> u32 {
        self.opts.codec_params.sample_rate.unwrap_or(0)
    }

    pub fn num_channels(&self) -> u16 {
        self.opts.codec_params.num_channels
    }

    pub fn bits_per_sample(&self) -> u16 {
        self.opts.codec_params.bits_per_sample.unwrap_or(0)
    }

    pub fn sample_format(&self) -> &SampleFormat {
        self.opts.codec_params.sample_format.as_ref().unwrap()
    }

    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.opts.metadata
    }
}

/// Decodificador de archivos WAV.
pub struct WavDecoder<R: Read + Seek + Debug> {
    reader: WavReader<R>,
}

impl<R: Read + Seek + Debug> WavDecoder<R> {
    /// Crea un nuevo decodificador WAV a partir de un `WavReader`.
    pub fn try_new(mut reader: WavReader<R>) -> Result<Self> {
        // Verifica que el formato de audio sea PCM (1)
        if reader.opts.codec_params.audio_format != Some(1) {
            return Err(Error::Static(
                "Unsupported audio format (only PCM is supported)",
            ));
        }

        // Mueve el lector a la posición de inicio de los datos de audio
        reader
            .reader
            .seek(SeekFrom::Start(reader.opts.data_start))?;

        Ok(Self { reader })
    }

    /// Devuelve un iterador sobre las muestras de audio decodificadas.
    pub fn samples(&mut self) -> SampleIterator<R> {
        let sample_format = *self.reader.sample_format();
        let num_channels = self.reader.num_channels();

        SampleIterator::new(
            &mut self.reader.reader,
            self.reader.opts.data_end,
            sample_format,
            num_channels,
        )
    }
}

/// Iterador sobre las muestras de audio decodificadas.
pub struct SampleIterator<'a, R: Read + Seek + Debug> {
    reader: &'a mut BufReader<R>,
    data_end: u64,
    sample_format: SampleFormat,
    num_channels: u16,
}

impl<'a, R: Read + Seek + Debug> SampleIterator<'a, R> {
    fn new(
        reader: &'a mut BufReader<R>,
        data_end: u64,
        sample_format: SampleFormat,
        num_channels: u16,
    ) -> Self {
        Self {
            reader,
            data_end,
            sample_format,
            num_channels,
        }
    }
}

impl<R: Read + Seek + Debug> Iterator for SampleIterator<'_, R> {
    type Item = Result<Vec<i32>>;

    fn next(&mut self) -> Option<Self::Item> {
        let pos = match self.reader.stream_position() {
            Ok(pos) => pos,
            Err(e) => return Some(Err(Error::Io(e))),
        };

        let pos = match self.reader.stream_position() {
            Ok(pos) => pos,
            Err(e) => return Some(Err(Error::Io(e))),
        };
        if pos >= self.data_end {
            return None;
        }

        let bytes_per_sample = self.sample_format.bytes_per_sample() as usize;
        let total_bytes = bytes_per_sample * self.num_channels as usize;

        let mut buffer = vec![0u8; total_bytes];
        match self.reader.read_exact(&mut buffer) {
            Ok(_) => {
                let mut samples = Vec::with_capacity(self.num_channels as usize);
                for ch in 0..self.num_channels {
                    let offset = ch as usize * bytes_per_sample;
                    let sample = match self.sample_format {
                        SampleFormat::Uint8 => {
                            // PCM sin signo de 8 bits, offset de 128
                            let s = buffer[offset] as i32 - 128;
                            s << 24 // Escalar a 32 bits
                        }
                        SampleFormat::Int16 => {
                            let s = i16::from_le_bytes([buffer[offset], buffer[offset + 1]]) as i32;
                            s << 16 // Escalar a 32 bits
                        }
                        SampleFormat::Int24 => {
                            let b0 = buffer[offset];
                            let b1 = buffer[offset + 1];
                            let b2 = buffer[offset + 2];
                            let s = ((b2 as i32) << 16) | ((b1 as i32) << 8) | (b0 as i32);
                            // Extender signo a 32 bits

                            if s & 0x800000 != 0 {
                                s | 0xFF000000u32 as i32
                            } else {
                                s
                            }
                        }
                        SampleFormat::Int32 => {
                            let s = i32::from_le_bytes([
                                buffer[offset],
                                buffer[offset + 1],
                                buffer[offset + 2],
                                buffer[offset + 3],
                            ]);
                            s
                        }
                    };
                    samples.push(sample);
                }
                Some(Ok(samples))
            }
            Err(e) => Some(Err(Error::Io(e))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // Helper function to create a simple WAV file in memory
    fn create_test_wav(
        sample_format: SampleFormat,
        num_channels: u16,
        sample_rate: u32,
    ) -> Vec<u8> {
        let bits_per_sample = sample_format.bytes_per_sample() * 8;
        let byte_rate = sample_rate * num_channels as u32 * sample_format.bytes_per_sample() as u32;
        let block_align = num_channels * sample_format.bytes_per_sample();

        let mut data = Vec::new();

        // RIFF header
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&0u32.to_le_bytes()); // Placeholder for chunk size
        data.extend_from_slice(b"WAVE");

        // fmt chunk
        data.extend_from_slice(b"fmt ");
        data.extend_from_slice(&16u32.to_le_bytes()); // Subchunk1Size
        data.extend_from_slice(&1u16.to_le_bytes()); // AudioFormat (PCM)
        data.extend_from_slice(&num_channels.to_le_bytes());
        data.extend_from_slice(&sample_rate.to_le_bytes());
        data.extend_from_slice(&byte_rate.to_le_bytes());
        data.extend_from_slice(&block_align.to_le_bytes());
        data.extend_from_slice(&bits_per_sample.to_le_bytes());

        // data chunk
        data.extend_from_slice(b"data");
        data.extend_from_slice(&0u32.to_le_bytes()); // Placeholder for data chunk size

        // Placeholder for audio data (we'll leave it empty for now)
        let data_chunk_size = 0u32;
        let riff_chunk_size = (data.len() - 8) as u32;

        // Update the placeholder values
        data[4..8].copy_from_slice(&riff_chunk_size.to_le_bytes());
        let len = data.len();
        data[(len - data_chunk_size as usize - 4)..(len - data_chunk_size as usize)]
            .copy_from_slice(&data_chunk_size.to_le_bytes());

        data
    }

    #[test]
    fn test_wav_reader_creation() {
        let wav_data = create_test_wav(SampleFormat::Int16, 2, 44100);
        let cursor = Cursor::new(wav_data);

        let reader = WavReader::try_new(cursor).expect("Failed to create WavReader");
        assert_eq!(reader.sample_rate(), 44100);
        assert_eq!(reader.num_channels(), 2);
        assert_eq!(reader.bits_per_sample(), 16);
        assert_eq!(reader.sample_format(), &SampleFormat::Int16);
    }

    #[test]
    fn test_wav_decoder_creation() {
        let wav_data = create_test_wav(SampleFormat::Int16, 2, 44100);
        let cursor = Cursor::new(wav_data);

        let reader = WavReader::try_new(cursor).expect("Failed to create WavReader");
        let decoder = WavDecoder::try_new(reader).expect("Failed to create WavDecoder");
        assert_eq!(decoder.reader.sample_rate(), 44100);
        assert_eq!(decoder.reader.num_channels(), 2);
        assert_eq!(decoder.reader.bits_per_sample(), 16);
        assert_eq!(decoder.reader.sample_format(), &SampleFormat::Int16);
    }

    #[test]
    fn test_sample_iterator_empty_data() {
        let wav_data = create_test_wav(SampleFormat::Int16, 2, 44100);
        let cursor = Cursor::new(wav_data);

        let reader = WavReader::try_new(cursor).expect("Failed to create WavReader");
        let mut decoder = WavDecoder::try_new(reader).expect("Failed to create WavDecoder");

        let mut samples = decoder.samples();
        assert!(samples.next().is_none());
    }

    #[test]
    fn test_invalid_riff_header() {
        let mut wav_data = create_test_wav(SampleFormat::Int16, 2, 44100);
        // Corrupt the RIFF header
        wav_data[0..4].copy_from_slice(b"FAIL");
        let cursor = Cursor::new(wav_data);

        let result = WavReader::try_new(cursor);
        assert!(result.is_err());
        match result {
            Err(Error::Static(msg)) => assert_eq!(msg, "Invalid RIFF header"),
            _ => panic!("Expected invalid RIFF header error"),
        }
    }

    #[test]
    fn test_invalid_wave_header() {
        let mut wav_data = create_test_wav(SampleFormat::Int16, 2, 44100);
        // Corrupt the WAVE header
        wav_data[8..12].copy_from_slice(b"FAIL");
        let cursor = Cursor::new(wav_data);

        let result = WavReader::try_new(cursor);
        assert!(result.is_err());
        match result {
            Err(Error::Static(msg)) => assert_eq!(msg, "Invalid WAVE header"),
            _ => panic!("Expected invalid WAVE header error"),
        }
    }

    #[test]
    fn test_unsupported_audio_format() {
        let mut wav_data = create_test_wav(SampleFormat::Int16, 2, 44100);
        // Change AudioFormat to an unsupported value (e.g., 2)
        wav_data[20..22].copy_from_slice(&2u16.to_le_bytes());
        let cursor = Cursor::new(wav_data);

        let reader = WavReader::try_new(cursor).expect("Failed to create WavReader");
        let result = WavDecoder::try_new(reader);
        assert!(result.is_err());
        match result {
            Err(Error::Static(msg)) => {
                assert_eq!(msg, "Unsupported audio format (only PCM is supported)")
            }
            _ => panic!("Expected unsupported audio format error"),
        }
    }

    #[test]
    fn test_read_samples() {
        // Create a WAV with a single sample per channel
        let sample_format = SampleFormat::Int16;
        let num_channels = 2;
        let sample_rate = 44100;
        let bits_per_sample = sample_format.bytes_per_sample() * 8;
        let byte_rate = sample_rate * num_channels as u32 * sample_format.bytes_per_sample() as u32;
        let block_align = num_channels * sample_format.bytes_per_sample();

        let mut data = Vec::new();

        // RIFF header
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&0u32.to_le_bytes()); // Placeholder for chunk size
        data.extend_from_slice(b"WAVE");

        // fmt chunk
        data.extend_from_slice(b"fmt ");
        data.extend_from_slice(&16u32.to_le_bytes()); // Subchunk1Size
        data.extend_from_slice(&1u16.to_le_bytes()); // AudioFormat (PCM)
        data.extend_from_slice(&num_channels.to_le_bytes());
        data.extend_from_slice(&sample_rate.to_le_bytes());
        data.extend_from_slice(&byte_rate.to_le_bytes());
        data.extend_from_slice(&block_align.to_le_bytes());
        data.extend_from_slice(&bits_per_sample.to_le_bytes());

        // data chunk
        data.extend_from_slice(b"data");
        let data_chunk_size = (sample_format.bytes_per_sample() * num_channels) as u32;
        data.extend_from_slice(&data_chunk_size.to_le_bytes());

        // Audio data: let's write sample values 1000 and -1000
        let sample1 = 1000i16.to_le_bytes();
        let sample2 = (-1000i16).to_le_bytes();
        data.extend_from_slice(&sample1);
        data.extend_from_slice(&sample2);

        // Update the RIFF chunk size
        let riff_chunk_size = (data.len() - 8) as u32;
        data[4..8].copy_from_slice(&riff_chunk_size.to_le_bytes());

        let cursor = Cursor::new(data);

        let reader = WavReader::try_new(cursor).expect("Failed to create WavReader");
        let mut decoder = WavDecoder::try_new(reader).expect("Failed to create WavDecoder");

        let mut samples = decoder.samples();

        if let Some(Ok(sample_vec)) = samples.next() {
            assert_eq!(sample_vec.len(), num_channels as usize);
            assert_eq!(sample_vec[0], 1000i32 << 16); // Shifted to 32 bits
            assert_eq!(sample_vec[1], (-1000i32) << 16);
        } else {
            panic!("Failed to read samples");
        }

        assert!(samples.next().is_none());
    }

    #[test]
    fn test_metadata_parsing() {
        // Create a WAV with a LIST chunk containing metadata
        let sample_format = SampleFormat::Int16;
        let num_channels = 2;
        let sample_rate = 44100;
        let bits_per_sample = sample_format.bytes_per_sample() * 8;
        let byte_rate = sample_rate * num_channels as u32 * sample_format.bytes_per_sample() as u32;
        let block_align = num_channels * sample_format.bytes_per_sample();

        let mut data = Vec::new();
        let data_lenght = data.len();
        // RIFF header
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&0u32.to_le_bytes()); // Placeholder for chunk size
        data.extend_from_slice(b"WAVE");

        // fmt chunk
        data.extend_from_slice(b"fmt ");
        data.extend_from_slice(&16u32.to_le_bytes()); // Subchunk1Size
        data.extend_from_slice(&1u16.to_le_bytes()); // AudioFormat (PCM)
        data.extend_from_slice(&num_channels.to_le_bytes());
        data.extend_from_slice(&sample_rate.to_le_bytes());
        data.extend_from_slice(&byte_rate.to_le_bytes());
        data.extend_from_slice(&block_align.to_le_bytes());
        data.extend_from_slice(&bits_per_sample.to_le_bytes());

        // LIST chunk
        data.extend_from_slice(b"LIST");
        let list_chunk_size = 4 + 4 + 4 + 4 + 5 + 1; // ListType + INFO + key + size + value + padding
        data.extend_from_slice(&(list_chunk_size as u32).to_le_bytes());
        data.extend_from_slice(b"INFO");
        data.extend_from_slice(b"INAM"); // Key for TrackTitle
        data.extend_from_slice(&5u32.to_le_bytes()); // Size of the value
        data.extend_from_slice(b"Test\0"); // Value with null terminator
        data.extend_from_slice(&0u8.to_le_bytes()); // Padding to even byte boundary

        // data chunk
        data.extend_from_slice(b"data");
        data.extend_from_slice(&0u32.to_le_bytes()); // Placeholder for data chunk size

        // Placeholder for audio data (empty)
        let data_chunk_size = 0u32;
        let riff_chunk_size = (data_lenght - 8) as u32;

        // Update placeholders
        data[4..8].copy_from_slice(&riff_chunk_size.to_le_bytes());
        data[(data_lenght - data_chunk_size as usize - 4)
            ..(data_lenght - data_chunk_size as usize)]
            .copy_from_slice(&data_chunk_size.to_le_bytes());

        let cursor = Cursor::new(data);

        let reader = WavReader::try_new(cursor).expect("Failed to create WavReader");
        assert_eq!(reader.metadata().len(), 1);
        assert_eq!(reader.metadata().get("INAM").unwrap(), "Test");
    }
}
