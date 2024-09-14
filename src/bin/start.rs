use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};

#[derive(Debug, thiserror::Error)]
enum Error {
    /// Represents a static error message.
    #[error("{0}")]
    Static(&'static str),
    /// Represents a generic error with a dynamic message.
    #[error("Generic {0}")]
    Generic(String),
    /// Represents an IO error, wrapped from std::io::Error.
    #[error(transparent)]
    Io(#[from] io::Error),
}

/// Specialized `Result` type for operations in this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Represents different sample formats.
#[derive(Debug)]
pub enum SampleFormat {
    Uint8,
    Int16,
    Int24,
    Float32,
}

/// Represents the time base (rational number as numerator and denominator).
#[derive(Debug, Default)]
pub struct TimeBase {
    pub numer: u32,
    pub denom: u32,
}

/// Represents codec parameters for the WAV file.
#[derive(Debug, Default)]
struct CodecParams {
    pub codec: Option<u32>,
    pub sample_rate: Option<u32>,
    pub time_base: Option<TimeBase>,
    pub num_frames: Option<u64>,
    pub start_ts: u64,
    pub sample_format: Option<SampleFormat>,
    pub bits_per_sample: Option<u32>,
    pub bits_per_coded_sample: Option<u32>,
    pub num_channels: u8,
    pub delay: Option<u32>,
    pub padding: Option<u32>,
    pub max_frames_per_packet: Option<u64>,
    pub packet_data_integrity: bool,
    pub frames_per_block: Option<u64>,
    pub extra_data: Option<Box<[u8]>>,
}

/// Represents the "fmt " chunk in a WAV file.
struct FormatChunk {
    audio_format: u16,
    num_channels: u16,
    sample_rate: u32,
    byte_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
}

/// Represents the "LIST" chunk in a WAV file.
struct ListChunk {
    list_type: [u8; 4],
    length: u32,
}

/// Represents the "fact" chunk in a WAV file.
struct FactChunk {
    num_samples: u32,
}

/// Represents the "data" chunk in a WAV file.
struct DataChunk {
    length: u32,
    data_position: u64,
}

/// Enum representing various WAV chunk types.
pub enum WaveChunk {
    Format(FormatChunk),
    List(ListChunk),
    Fact(FactChunk),
    Data(DataChunk),
}

/// Enum for identifying chunk types based on their ID.
pub enum ChunkType {
    Format,
    List,
    Fact,
    Data,
}

impl ChunkType {
    /// Converts a 4-byte ID into a corresponding `ChunkType`.
    pub fn from_id(id: &[u8; 4]) -> Option<Self> {
        match id {
            b"fmt " => Some(Self::Format),
            b"LIST" => Some(Self::List),
            b"fact" => Some(Self::Fact),
            b"data" => Some(Self::Data),
            _ => None,
        }
    }
}

/// Enum representing byte order types.
pub enum ByteOrder {
    LittleEndian,
    BigEndian,
}

/// A buffered source stream for reading data from a WAV file.
struct SourceStream<R: Read + Seek> {
    reader: R,
    abs_pos: u64,
}

impl<R: Read + Seek> SourceStream<R> {
    /// Creates a new `SourceStream` with the given reader.
    pub fn new(reader: R) -> Self {
        Self { reader, abs_pos: 0 }
    }

    /// Reads exactly `N` bytes from the source stream.
    pub fn read_exact<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut result = [0u8; N];
        self.reader.read_exact(&mut result)?;
        self.abs_pos += N as u64;
        Ok(result)
    }

    /// Reads a `u16` in little-endian format.
    pub fn read_u16_le(&mut self) -> Result<u16> {
        let bytes = self.read_exact::<2>()?;
        Ok(u16::from_le_bytes(bytes))
    }

    /// Reads a `u32` in little-endian format.
    pub fn read_u32_le(&mut self) -> Result<u32> {
        let bytes = self.read_exact::<4>()?;
        Ok(u32::from_le_bytes(bytes))
    }

    /// Reads a `u32` in big-endian format.
    pub fn read_u32_be(&mut self) -> Result<u32> {
        let bytes = self.read_exact::<4>()?;
        Ok(u32::from_be_bytes(bytes))
    }

    /// Seeks to a specific position in the stream.
    pub fn seek(&mut self, pos: SeekFrom) -> Result<u64> {
        let new_pos = self.reader.seek(pos)?;
        self.abs_pos = new_pos;
        Ok(new_pos)
    }

    /// Returns the current position in the stream.
    pub fn position(&self) -> u64 {
        self.abs_pos
    }
}

/// Parses chunks from a WAV file.
struct ChunkParser<'a, R: Read + Seek> {
    source_stream: &'a mut SourceStream<R>,
    byte_order: ByteOrder,
    cursor: usize,
    length: usize,
}

impl<'a, R: Read + Seek> ChunkParser<'a, R> {
    /// Creates a new chunk parser with the given source stream and byte order.
    pub fn new(source: &'a mut SourceStream<R>, byte_order: ByteOrder, length: usize) -> Self {
        Self {
            source_stream: source,
            byte_order,
            cursor: 0,
            length,
        }
    }

    /// Aligns the cursor to the next 2-byte boundary, if needed.
    fn align(&mut self) -> Result<()> {
        if self.cursor & 1 != 0 {
            self.source_stream.seek(SeekFrom::Current(1))?;
            self.cursor += 1;
        }
        Ok(())
    }

    /// Moves the cursor by the specified number of bytes.
    fn move_cursor(&mut self, n: usize) -> Result<()> {
        self.source_stream.seek(SeekFrom::Current(n as i64))?;
        self.cursor += n;
        Ok(())
    }

    /// Iterates over each chunk and applies the given function to each.
    pub fn for_each_chunk<F>(&mut self, mut f: F) -> Result<()>
    where
        F: FnMut(WaveChunk) -> Result<()>,
    {
        while self.cursor + 8 <= self.length {
            self.align()?;

            // Read the chunk ID.
            let chunk_id = self.source_stream.read_exact::<4>()?;
            self.cursor += 4;

            // Read the chunk size.
            let chunk_size = match self.byte_order {
                ByteOrder::LittleEndian => self.source_stream.read_u32_le()?,
                ByteOrder::BigEndian => self.source_stream.read_u32_be()?,
            };
            self.cursor += 4;

            // Process the chunk based on its ID.
            let chunk = match &chunk_id {
                b"fmt " => self.parse_format_chunk(chunk_size)?,
                b"data" => self.parse_data_chunk(chunk_size)?,
                b"LIST" => self.parse_list_chunk(chunk_size)?,
                b"fact" => self.parse_fact_chunk(chunk_size)?,
                _ => {
                    self.source_stream
                        .seek(SeekFrom::Current(chunk_size as i64))?;
                    self.cursor += chunk_size as usize;
                    continue;
                }
            };

            f(chunk)?;
        }

        Ok(())
    }

    /// Parses a "fmt " chunk.
    fn parse_format_chunk(&mut self, chunk_size: u32) -> Result<WaveChunk> {
        let audio_format = self.source_stream.read_u16_le()?;
        let num_channels = self.source_stream.read_u16_le()?;
        let sample_rate = self.source_stream.read_u32_le()?;
        let byte_rate = self.source_stream.read_u32_le()?;
        let block_align = self.source_stream.read_u16_le()?;
        let bits_per_sample = self.source_stream.read_u16_le()?;
        self.cursor += 16;

        if chunk_size > 16 {
            let extra_size = (chunk_size - 16) as i64;
            self.source_stream.seek(SeekFrom::Current(extra_size))?;
            self.cursor += extra_size as usize;
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

    /// Parses a "data" chunk.
    fn parse_data_chunk(&mut self, chunk_size: u32) -> Result<WaveChunk> {
        let data_position = self.source_stream.position();
        self.source_stream
            .seek(SeekFrom::Current(chunk_size as i64))?;
        self.cursor += chunk_size as usize;

        Ok(WaveChunk::Data(DataChunk {
            length: chunk_size,
            data_position,
        }))
    }

    /// Parses a "LIST" chunk.
    fn parse_list_chunk(&mut self, chunk_size: u32) -> Result<WaveChunk> {
        let list_type = self.source_stream.read_exact::<4>()?;
        self.cursor += 4;

        let remaining_size = chunk_size as usize - 4;
        self.source_stream
            .seek(SeekFrom::Current(remaining_size as i64))?;
        self.cursor += remaining_size;

        Ok(WaveChunk::List(ListChunk {
            list_type,
            length: chunk_size,
        }))
    }

    /// Parses a "fact" chunk.
    fn parse_fact_chunk(&mut self, chunk_size: u32) -> Result<WaveChunk> {
        let num_samples = self.source_stream.read_u32_le()?;
        self.cursor += 4;

        if chunk_size > 4 {
            let extra_size = (chunk_size - 4) as i64;
            self.source_stream.seek(SeekFrom::Current(extra_size))?;
            self.cursor += extra_size as usize;
        }

        Ok(WaveChunk::Fact(FactChunk { num_samples }))
    }
}

/// WAV file reader.
struct WavReader<R: Read + Seek> {
    source_stream: SourceStream<R>,
    codec_params: CodecParams,
    data_chunk: Option<DataChunk>,
}

impl<R: Read + Seek> WavReader<R> {
    const RIFF_HEADER: [u8; 4] = *b"RIFF";
    const WAVE_HEADER: [u8; 4] = *b"WAVE";

    /// Tries to create a new `WavReader` by parsing the WAV file headers.
    pub fn try_new(mut source_stream: SourceStream<R>) -> Result<Self> {
        let riff_header = source_stream.read_exact::<4>()?;
        if riff_header != Self::RIFF_HEADER {
            return Err(Error::Static("Invalid RIFF header"));
        }

        let chunk_size = source_stream.read_u32_le()?;
        let wave_header = source_stream.read_exact::<4>()?;
        if wave_header != Self::WAVE_HEADER {
            return Err(Error::Static("Invalid WAVE header"));
        }

        let mut codec_params = CodecParams::default();
        let mut data_chunk = None;

        let mut parser = ChunkParser::new(
            &mut source_stream,
            ByteOrder::LittleEndian,
            chunk_size as usize - 4,
        );

        parser.for_each_chunk(|chunk| {
            match chunk {
                WaveChunk::Format(format_chunk) => {
                    codec_params.sample_rate = Some(format_chunk.sample_rate);
                    codec_params.num_channels = format_chunk.num_channels as u8;
                    codec_params.bits_per_sample = Some(format_chunk.bits_per_sample as u32);
                }
                WaveChunk::Data(dc) => {
                    data_chunk = Some(dc);
                }
                _ => {}
            }
            Ok(())
        })?;

        Ok(Self {
            source_stream,
            codec_params,
            data_chunk,
        })
    }

    /// Reads the audio data from the DataChunk.
    pub fn read_audio_data(&mut self) -> Result<Vec<u8>> {
        if let Some(ref data_chunk) = self.data_chunk {
            let data_position = SeekFrom::Start(data_chunk.data_position);
            self.source_stream.seek(data_position)?;

            let mut audio_data = vec![0u8; data_chunk.length as usize];
            self.source_stream.reader.read_exact(&mut audio_data)?;

            return Ok(audio_data);
        }

        Err(Error::Static("Data chunk not found"))
    }
}

fn main() -> Result<()> {
    let file = File::open("dulce_carita.wav")?;
    let source_stream = SourceStream::new(file);
    let mut wav_reader = WavReader::try_new(source_stream)?;

    // Read the audio data.
    let audio_data = wav_reader.read_audio_data()?;
    // Process `audio_data` as needed.

    Ok(())
}
