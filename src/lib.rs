use std::{
    collections::HashMap,
    fmt::Debug,
    io::{self, BufReader, Read, Seek, SeekFrom},
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
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

impl CodecParams {
    /// Creates a new `CodecParams` with default values.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_codec(&mut self, codec: u32) -> &mut Self {
        self.codec = Some(codec);
        self
    }

    pub fn set_sample_rate(&mut self, sample_rate: u32) -> &mut Self {
        self.sample_rate = Some(sample_rate);
        self
    }

    pub fn set_time_base(&mut self, numer: u32, denom: u32) -> &mut Self {
        self.time_base = Some(TimeBase { numer, denom });
        self
    }

    pub fn set_num_frames(&mut self, num_frames: u64) -> &mut Self {
        self.num_frames = Some(num_frames);
        self
    }

    pub fn set_start_ts(&mut self, start_ts: u64) -> &mut Self {
        self.start_ts = start_ts;
        self
    }

    pub fn set_sample_format(&mut self, sample_format: SampleFormat) -> &mut Self {
        self.sample_format = Some(sample_format);
        self
    }

    pub fn set_bits_per_sample(&mut self, bits_per_sample: u16) -> &mut Self {
        self.bits_per_sample = Some(bits_per_sample);
        self
    }

    pub fn set_bits_per_coded_sample(&mut self, bits_per_coded_sample: u32) -> &mut Self {
        self.bits_per_coded_sample = Some(bits_per_coded_sample);
        self
    }

    pub fn set_num_channels(&mut self, num_channels: u16) -> &mut Self {
        self.num_channels = num_channels;
        self
    }

    pub fn set_delay(&mut self, delay: u32) -> &mut Self {
        self.delay = Some(delay);
        self
    }

    pub fn set_padding(&mut self, padding: u32) -> &mut Self {
        self.padding = Some(padding);
        self
    }

    pub fn set_max_frames_per_packet(&mut self, max_frames_per_packet: u64) -> &mut Self {
        self.max_frames_per_packet = Some(max_frames_per_packet);
        self
    }

    pub fn set_packet_data_integrity(&mut self, packet_data_integrity: bool) -> &mut Self {
        self.packet_data_integrity = packet_data_integrity;
        self
    }

    pub fn set_frames_per_block(&mut self, frames_per_block: u64) -> &mut Self {
        self.frames_per_block = Some(frames_per_block);
        self
    }

    pub fn set_extra_data(&mut self, extra_data: Box<[u8]>) -> &mut Self {
        self.extra_data = Some(extra_data);
        self
    }
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

impl FormatChunk {
    /// Returns the number of bytes per sample.
    pub fn bytes_per_sample(&self) -> u16 {
        self.bits_per_sample / 8
    }
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
    Unknown,
}

impl ChunkType {
    /// Converts a 4-byte ID into a corresponding `ChunkType`.
    pub fn from_id(id: &[u8; 4]) -> Self {
        use ChunkType::*;
        match id {
            b"fmt " => Format,
            b"LIST" => List,
            b"fact" => Fact,
            b"data" => Data,
            _ => Unknown,
        }
    }
}

/// Enum representing byte order types.
pub enum ByteOrder {
    LittleEndian,
    BigEndian,
}

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
struct ChunkParser<'a, R: Read + Seek + Debug> {
    source_stream: &'a mut SourceStream<R>,
    cursor: usize,
    length: usize,
}

impl<'a, R: Read + Seek + Debug> ChunkParser<'a, R> {
    /// Creates a new chunk parser with the given source stream and byte order.
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

            // Read the chunk ID and size.
            let chunk_id = self.source_stream.read_exact::<4>()?;
            let chunk_size = self.source_stream.read_u32_le()?;
            let chunk_size_usize = chunk_size as usize;
            self.cursor += 8;

            // Special case where the chunk size is `u32::MAX` and matches `self.length`.
            let is_special_case = self.length == chunk_size_usize && chunk_size == u32::MAX;

            // Check if the chunk_size exceeds the remaining bytes, excluding the special case.
            if !is_special_case && self.length - self.cursor < chunk_size_usize {
                return Err(Error::Static("Chunk size exceeds the remaining length"));
            }

            self.cursor = self.cursor.saturating_add(chunk_size_usize);

            // Process the chunk based on its ID.
            let chunk = match ChunkType::from_id(&chunk_id) {
                ChunkType::Format => self.parse_format_chunk(chunk_size)?,
                ChunkType::Data => self.parse_data_chunk(chunk_size)?,
                ChunkType::List => self.parse_list_chunk(chunk_size)?,
                ChunkType::Fact => self.parse_fact_chunk(chunk_size)?,
                ChunkType::Unknown => {
                    self.source_stream
                        .seek(SeekFrom::Current(chunk_size as i64))?;
                    self.cursor += chunk_size_usize;
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

const MAX_FRAMES_PER_PACKET: u64 = 1024;

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

    pub fn set_block_size(&mut self, block_size: u64) -> &mut Self {
        self.block_size = block_size;
        self
    }

    pub fn set_frames_per_block(&mut self, frames_per_block: u64) -> &mut Self {
        self.frames_per_block = frames_per_block;
        self
    }

    pub fn set_max_blocks_per_packet(&mut self, max_blocks_per_packet: u64) -> &mut Self {
        self.max_blocks_per_packet = max_blocks_per_packet;
        self
    }
}

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
            ..Default::default()
        }
    }
}

impl WavReaderOptions {
    fn set_codec_params(&mut self, codec_params: CodecParams) -> &mut Self {
        self.codec_params = codec_params;
        self
    }

    fn set_metadata(&mut self, key: String, value: String) -> &mut Self {
        self.metadata.insert(key, value);
        self
    }

    fn set_packet_info(&mut self, packet_info: PacketInfo) -> &mut Self {
        self.packet_info = packet_info;
        self
    }
}

/// WAV file reader.
pub struct WavReader<R: Read + Seek + Debug> {
    source_stream: SourceStream<R>,
    opts: WavReaderOptions,
}

impl<R: Read + Seek + Debug> WavReader<R> {
    const RIFF_HEADER: [u8; 4] = *b"RIFF";
    const WAVE_HEADER: [u8; 4] = *b"WAVE";
    const INFO_HEADER: [u8; 4] = *b"INFO";

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

        let chunk_size = source_stream.read_u32_le()?;
        let wave_header = source_stream.read_exact::<4>()?;
        if wave_header != Self::WAVE_HEADER {
            return Err(Error::Static("Invalid WAVE header"));
        }

        let mut opts = WavReaderOptions::default();
        let mut parser = ChunkParser::new(&mut source_stream, chunk_size as usize - 4);

        parser.for_each_chunk(|chunk| {
            use WaveChunk::*;
            match chunk {
                Format(format) => Self::add_fmt_data(&mut opts, format),
                Data(data) => Self::add_data_data(&mut opts, data),
                Fact(fact) => Self::add_fact_data(&mut opts, fact),
                List(list) => Self::add_list_data(&mut opts, list),
            }
            Ok(())
        })?;

        Ok(Self::new(source_stream, opts))
    }

    fn add_fmt_data(source: &mut WavReaderOptions, chunk: FormatChunk) {
        let packet_info = source.packet_info.set_block_size(chunk.block_align as u64);
        source
            .codec_params
            .set_frames_per_block(packet_info.frames_per_block)
            .set_max_frames_per_packet(packet_info.max_blocks_per_packet);
    }

    fn add_fact_data(source: &mut WavReaderOptions, chunk: FactChunk) {
        source.codec_params.set_num_frames(chunk.num_samples as u64);
    }

    fn add_data_data(source: &mut WavReaderOptions, chunk: DataChunk) {
        source.data_start = chunk.data_position;
        source.data_end = chunk.data_position + chunk.length as u64;

        let packet_info = &source.packet_info;

        if packet_info.block_size != 0 {
            let num_frames =
                chunk.length as u64 / (packet_info.block_size * packet_info.frames_per_block);
            source.codec_params.set_num_frames(num_frames);
        }
    }

    fn add_list_data(source: &mut WavReaderOptions, chunk: ListChunk) {}
}

/// Información de tipo basada en claves.
enum TypeInfo {
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

impl TypeInfo {
    /// Convierte un arreglo de 4 bytes a `TypeInfo`.
    fn from_bytes(key: &[u8; 4]) -> Self {
        match key {
            b"ages" => TypeInfo::Rating,
            b"cmnt" | b"comm" | b"icmt" => TypeInfo::Comment,
            b"dtim" | b"idit" => TypeInfo::OriginalDate,
            b"genr" | b"ignr" | b"isgn" => TypeInfo::Genre,
            b"iart" => TypeInfo::Artist,
            b"icop" => TypeInfo::Copyright,
            b"icrd" | b"year" => TypeInfo::Date,
            b"ienc" | b"itch" => TypeInfo::EncodedBy,
            b"ieng" => TypeInfo::Engineer,
            b"ifrm" => TypeInfo::TrackTotal,
            b"ilng" | b"lang" => TypeInfo::Language,
            b"imus" => TypeInfo::Composer,
            b"inam" | b"titl" => TypeInfo::TrackTitle,
            b"iprd" => TypeInfo::Album,
            b"ipro" => TypeInfo::Producer,
            b"iprt" | b"trck" | b"prt1" | b"prt2" => TypeInfo::TrackNumber,
            b"isft" => TypeInfo::Encoder,
            b"isrf" => TypeInfo::MediaFormat,
            b"iwri" => TypeInfo::Writer,
            b"torg" => TypeInfo::Label,
            b"tver" => TypeInfo::Version,
            _ => TypeInfo::Unknown,
        }
    }
}
