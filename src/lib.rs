use std::{
    collections::HashMap,
    fmt::Debug,
    io::{self, BufReader, Read, Seek, SeekFrom},
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Static(&'static str),
    #[error("Error genérico: {0}")]
    Generic(String),
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
    Int32,
}

/// Represents the time base (rational number as numerator and denominator).
#[derive(Debug, Default)]
pub struct TimeBase {
    pub numer: u32,
    pub denom: u32,
}

/// Represents codec parameters for the WAV file.
#[derive(Debug, Default)]
pub struct CodecParams {
    pub codec: Option<u32>,
    pub sample_rate: Option<u32>,
    pub time_base: Option<TimeBase>,
    pub num_frames: Option<u64>,
    pub start_ts: u64,
    pub sample_format: Option<SampleFormat>,
    pub bits_per_sample: Option<u16>,
    pub bits_per_coded_sample: Option<u32>,
    pub num_channels: u16,
    pub delay: Option<u32>,
    pub padding: Option<u32>,
    pub max_frames_per_packet: Option<u64>,
    pub packet_data_integrity: bool,
    pub frames_per_block: Option<u64>,
    pub extra_data: Option<Box<[u8]>>,
}

/// Represents the "fmt " chunk in a WAV file.
pub struct FormatChunk {
    audio_format: u16,
    num_channels: u16,
    sample_rate: u32,
    byte_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
}

impl FormatChunk {
    /// Returns the number of bytes per sample.
    pub fn bytes_per_sample(&self) -> u16 {
        self.bits_per_sample / 8
    }
}

/// Represents the "LIST" chunk in a WAV file.
pub struct ListChunk {
    list_type: [u8; 4],
    length: u32,
    tags: Vec<Tag>,
}

/// Represents the "fact" chunk in a WAV file.
pub struct FactChunk {
    num_samples: u32,
}

/// Represents the "data" chunk in a WAV file.
pub struct DataChunk {
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
    Unknown,
}

impl ChunkType {
    /// Converts a 4-byte ID into a corresponding `ChunkType`.
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

/// A `Tag` encapsulates a key-value pair of metadata.
#[derive(Clone, Debug)]
pub struct Tag {
    pub tag_type: Option<TagType>,
    pub key: String,
    pub value: String,
}

/// Byte order types (not used in the refactored code, so removed).

const BUFFER_SIZE: usize = 1024 * 16;

/// A buffered source stream for reading data from a WAV file.
#[derive(Debug)]
pub struct SourceStream<R: Read + Seek + Debug> {
    reader: BufReader<R>,
    abs_pos: u64,
}

impl<R: Read + Seek + Debug> SourceStream<R> {
    /// Creates a new `SourceStream` with the given reader.
    pub fn new(reader: R) -> Self {
        let reader = BufReader::with_capacity(BUFFER_SIZE, reader);
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
struct ChunkParser<'a, R: Read + Seek + Debug> {
    source_stream: &'a mut SourceStream<R>,
    cursor: usize,
    length: usize,
}

impl<'a, R: Read + Seek + Debug> ChunkParser<'a, R> {
    /// Creates a new chunk parser with the given source stream and length.
    pub fn new(source_stream: &'a mut SourceStream<R>, length: usize) -> Self {
        Self {
            source_stream,
            cursor: 0,
            length,
        }
    }

    /// Aligns the cursor to the next 2-byte boundary, if needed.
    fn align(&mut self) -> Result<()> {
        if self.cursor & 1 != 0 {
            self.skip_bytes(1)?;
        }
        Ok(())
    }

    /// Moves the cursor by the specified number of bytes.
    fn skip_bytes(&mut self, n: usize) -> Result<()> {
        self.source_stream.seek(SeekFrom::Current(n as i64))?;
        self.cursor += n;
        Ok(())
    }

    fn read_u16_le(&mut self) -> Result<u16> {
        let bytes = self.read_exact::<2>()?;
        Ok(u16::from_le_bytes(bytes))
    }

    fn read_u32_le(&mut self) -> Result<u32> {
        let bytes = self.read_exact::<4>()?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_exact<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut result = [0u8; N];
        self.source_stream.reader.read_exact(&mut result)?;
        self.cursor += N;
        Ok(result)
    }

    fn read_bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut buffer = vec![0; n];
        self.source_stream.reader.read_exact(&mut buffer)?;
        self.cursor += n;
        Ok(buffer)
    }

    /// Iterates over each chunk and applies the given function to each.
    pub fn parse_chunks<F>(&mut self, mut f: F) -> Result<()>
    where
        F: FnMut(WaveChunk) -> Result<()>,
    {
        while self.cursor + 8 <= self.length {
            // Read the chunk ID and size.
            let chunk_id = self.read_exact::<4>()?;
            let chunk_size = self.read_u32_le()?;
            let chunk_size_usize = chunk_size as usize;

            // Check if the chunk_size exceeds the remaining bytes.
            if self.length - self.cursor < chunk_size_usize {
                return Err(Error::Static("Chunk size exceeds the remaining length"));
            }

            // Process the chunk based on its ID.
            let chunk = match ChunkType::from_id(&chunk_id) {
                ChunkType::Format => self.parse_format_chunk(chunk_size_usize)?,
                ChunkType::Data => self.parse_data_chunk(chunk_size_usize)?,
                ChunkType::List => self.parse_list_chunk(chunk_size_usize)?,
                ChunkType::Fact => self.parse_fact_chunk(chunk_size_usize)?,
                ChunkType::Unknown => {
                    self.skip_bytes(chunk_size_usize)?;
                    continue;
                }
            };

            f(chunk)?;
            self.align()?;
        }

        Ok(())
    }

    /// Parses a "fmt " chunk.
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

    /// Parses a "data" chunk.
    fn parse_data_chunk(&mut self, chunk_size: usize) -> Result<WaveChunk> {
        let data_position = self.source_stream.position();
        self.cursor += chunk_size;
        Ok(WaveChunk::Data(DataChunk {
            length: chunk_size as u32,
            data_position,
        }))
    }

    /// Parses a "LIST" chunk.
    fn parse_list_chunk(&mut self, chunk_size: usize) -> Result<WaveChunk> {
        let list_type = self.read_exact::<4>()?;
        let remaining_size = chunk_size - 4;
        let mut tags = Vec::new();

        if &list_type == b"INFO" {
            self.parse_info_chunk(remaining_size, &mut tags)?;
        } else {
            self.skip_bytes(remaining_size)?;
        }

        Ok(WaveChunk::List(ListChunk {
            list_type,
            length: chunk_size as u32,
            tags,
        }))
    }

    /// Parses the "INFO" chunk which is a subchunk of the "LIST" chunk.
    fn parse_info_chunk(&mut self, mut remaining_size: usize, tags: &mut Vec<Tag>) -> Result<()> {
        while remaining_size >= 8 {
            // Read the key and size of the tag
            let tag_key = self.read_exact::<4>()?;
            let tag_size = self.read_u32_le()? as usize;
            remaining_size -= 8;

            // Read the value of the tag
            let tag_value_bytes = self.read_bytes(tag_size)?;
            let tag_value = String::from_utf8_lossy(&tag_value_bytes)
                .trim_end_matches(char::from(0))
                .to_string();
            remaining_size -= tag_size;

            // Align to even byte if necessary
            if tag_size % 2 != 0 {
                self.skip_bytes(1)?;
                remaining_size -= 1;
            }

            // Add the tag to the vector
            tags.push(Tag {
                tag_type: Some(TagType::from_bytes(&tag_key)),
                key: String::from_utf8_lossy(&tag_key).into_owned(),
                value: tag_value,
            });
        }
        Ok(())
    }

    /// Parses a "fact" chunk.
    fn parse_fact_chunk(&mut self, chunk_size: usize) -> Result<WaveChunk> {
        let num_samples = self.read_u32_le()?;
        if chunk_size > 4 {
            self.skip_bytes(chunk_size - 4)?;
        }
        Ok(WaveChunk::Fact(FactChunk { num_samples }))
    }
}

const MAX_FRAMES_PER_PACKET: u64 = 1024;

#[derive(Debug)]
pub struct PacketInfo {
    pub block_size: u64,
    pub frames_per_block: u64,
    pub max_blocks_per_packet: u64,
}

impl PacketInfo {
    pub fn new(frame_len: u16) -> Self {
        Self {
            frames_per_block: 1,
            block_size: frame_len as u64,
            max_blocks_per_packet: MAX_FRAMES_PER_PACKET,
        }
    }
}

#[derive(Debug)]
struct WavReaderOptions {
    codec_params: CodecParams,
    metadata: HashMap<String, String>,
    packet_info: PacketInfo,
    data_start: u64,
    data_end: u64,
}

impl Default for WavReaderOptions {
    fn default() -> Self {
        Self {
            codec_params: CodecParams::default(),
            metadata: HashMap::new(),
            packet_info: PacketInfo::new(0),
            data_start: 0,
            data_end: 0,
        }
    }
}

/// WAV file reader.
#[derive(Debug)]
pub struct WavReader<R: Read + Seek + Debug> {
    source_stream: SourceStream<R>,
    opts: WavReaderOptions,
}

impl<R: Read + Seek + Debug> WavReader<R> {
    const RIFF_HEADER: [u8; 4] = *b"RIFF";
    const WAVE_HEADER: [u8; 4] = *b"WAVE";

    fn new(source_stream: SourceStream<R>, opts: WavReaderOptions) -> Self {
        Self {
            source_stream,
            opts,
        }
    }

    /// Tries to create a new `WavReader` by parsing the WAV file headers.
    pub fn try_new(mut source_stream: SourceStream<R>) -> Result<Self> {
        let riff_header = source_stream.read_exact::<4>()?;
        if riff_header != Self::RIFF_HEADER {
            return Err(Error::Static("Invalid RIFF header"));
        }

        let chunk_size = source_stream.read_u32_le()? as usize;
        let wave_header = source_stream.read_exact::<4>()?;
        if wave_header != Self::WAVE_HEADER {
            return Err(Error::Static("Invalid WAVE header"));
        }

        let mut options = WavReaderOptions::default();
        let mut parser = ChunkParser::new(&mut source_stream, chunk_size);

        parser.parse_chunks(|chunk| match chunk {
            WaveChunk::Format(format) => Self::handle_format_chunk(&mut options, format),
            WaveChunk::Data(data) => Self::handle_data_chunk(&mut options, data),
            WaveChunk::Fact(fact) => Self::handle_fact_chunk(&mut options, fact),
            WaveChunk::List(list) => Self::handle_list_chunk(&mut options, list),
        })?;

        Ok(Self::new(source_stream, options))
    }

    fn handle_format_chunk(options: &mut WavReaderOptions, chunk: FormatChunk) -> Result<()> {
        options.packet_info.block_size = chunk.block_align as u64;
        options.codec_params.sample_rate = Some(chunk.sample_rate);
        options.codec_params.num_channels = chunk.num_channels;
        options.codec_params.bits_per_sample = Some(chunk.bits_per_sample);
        options.codec_params.frames_per_block = Some(options.packet_info.frames_per_block);
        options.codec_params.max_frames_per_packet =
            Some(options.packet_info.max_blocks_per_packet);

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

        if options.packet_info.block_size != 0 {
            let num_frames = chunk.length as u64
                / (options.packet_info.block_size * options.packet_info.frames_per_block);
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
}

/// Tag type based on keys.
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
    /// Converts a 4-byte array to `TagType`.
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
