use std::arch::x86_64::*;
use std::collections::HashMap;
use std::io::{self, BufReader, Read, Seek, SeekFrom};

use aligned_vec::{avec_rt, AVec, RuntimeAlign};
use bytemuck;
use enum_dispatch::enum_dispatch;
pub use fallible_streaming_iterator::FallibleStreamingIterator;

/// Number of samples per packet.
const PACK_SIZE: usize = 1024 * 8;
/// Maximum number of metadata tags to parse to prevent OOM attacks.
const MAX_TAGS: usize = 1_024;

// Validations for PACK_SIZE
const _: () = assert!(PACK_SIZE % 32 == 0, "must be multiple of 32 for uint8");
const _: () = assert!(PACK_SIZE % 16 == 0, "must be multiple of 16 for int16");
const _: () = assert!(PACK_SIZE % 8 == 0, "must be multiple of 8 for int32");

/// Specialized Result type for this crate's operations.
pub type Result<T> = std::result::Result<T, Error>;

/// General errors for this crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Static(&'static str),
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Utf8(#[from] std::str::Utf8Error),
    #[error("Arithmetic overflow detected during parsing.")]
    Overflow,
}

/// Represents different sample formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    Uint8,
    Int16,
    Int24,
    Int32,
}

impl SampleFormat {
    /// Returns the number of bytes per sample.
    #[inline]
    pub fn bytes_per_sample(&self) -> u16 {
        match self {
            SampleFormat::Uint8 => 1,
            SampleFormat::Int16 => 2,
            SampleFormat::Int24 => 3,
            SampleFormat::Int32 => 4,
        }
    }
}

/// Codec parameters for the WAV file.
#[derive(Debug, Default)]
struct CodecParams {
    pub sample_rate: Option<u32>,
    pub num_frames: Option<u64>,
    pub sample_format: Option<SampleFormat>,
    pub bits_per_sample: Option<u16>,
    pub num_channels: u16,
    pub block_align: Option<u16>,
    pub audio_format: Option<u16>,
    pub byte_rate: Option<u32>,
}

/// A Tag holds a key-value pair of metadata.
#[derive(Clone, Debug)]
pub struct Tag {
    pub tag_type: Option<TagType>,
    pub key: String,
    pub value: String,
}

/// Known tag types.
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
    /// Converts a 4-byte array into a TagType.
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

/// Represents the "fmt " chunk in a WAV file.
pub struct FormatChunk {
    audio_format: u16,
    num_channels: u16,
    sample_rate: u32,
    byte_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
}

/// Represents the "LIST" chunk in a WAV file.
pub struct ListChunk {
    _list_type: [u8; 4],
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

/// Enum representing several WAV chunk types.
pub enum WaveChunk {
    Format(FormatChunk),
    List(ListChunk),
    Fact(FactChunk),
    Data(DataChunk),
}

/// Enum identifying chunk types by their IDs.
pub enum ChunkType {
    Format,
    List,
    Fact,
    Data,
    Unknown,
}

impl ChunkType {
    /// Converts a 4-byte ID into the corresponding ChunkType.
    #[inline(always)]
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

/// A parser that processes WAV chunks.
struct ChunkParser<'a, R: Read + Seek> {
    reader: &'a mut BufReader<R>,
    cursor: u64,
    length: u64,
    file_len: u64,
}

impl<'a, R: Read + Seek> ChunkParser<'a, R> {
    /// Creates a new chunk parser with the given reader and length.
    pub fn new(reader: &'a mut BufReader<R>, length: u64, file_len: u64) -> Self {
        Self {
            reader,
            length,
            cursor: 0,
            file_len,
        }
    }

    /// Moves the cursor forward by `n` bytes.
    fn skip_bytes(&mut self, n: u64) -> Result<()> {
        self.reader.seek(SeekFrom::Current(n as i64))?;
        self.cursor = self.cursor.checked_add(n).ok_or(Error::Overflow)?;
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
        self.cursor = self.cursor.checked_add(N as u64).ok_or(Error::Overflow)?;
        Ok(result)
    }

    fn read_bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut buffer = vec![0; n];
        self.reader.read_exact(&mut buffer)?;
        self.cursor = self.cursor.checked_add(n as u64).ok_or(Error::Overflow)?;
        Ok(buffer)
    }

    /// Iterates over each chunk, applying the provided function to each.
    pub fn parse_chunks<F>(&mut self, mut f: F) -> Result<()>
    where
        F: FnMut(WaveChunk) -> Result<()>,
    {
        while self.cursor.checked_add(8).ok_or(Error::Overflow)? <= self.length {
            let chunk_id = self.read_exact::<4>()?;
            let chunk_size = self.read_u32_le()? as u64;
            let needs_alignment = chunk_size % 2 != 0;

            if self.length < self.cursor.checked_add(chunk_size).ok_or(Error::Overflow)? {
                return Err(Error::Static(
                    "Chunk size exceeds the remaining RIFF length",
                ));
            }

            if chunk_size == 0 {
                // CORRECTION #8: Handle padding byte for empty chunks
                if needs_alignment {
                    self.skip_bytes(1)?;
                }
                continue;
            }

            let chunk_start_pos = self.reader.stream_position()?;

            let chunk_result = match ChunkType::from_id(&chunk_id) {
                ChunkType::Format => self.parse_format_chunk(chunk_size as usize),
                ChunkType::Data => self.parse_data_chunk(chunk_size as usize),
                ChunkType::List => self.parse_list_chunk(chunk_size as usize),
                ChunkType::Fact => self.parse_fact_chunk(chunk_size as usize),
                ChunkType::Unknown => {
                    self.skip_bytes(chunk_size)?;
                    Ok(None)
                }
            };

            match chunk_result {
                Ok(Some(chunk)) => {
                    // Calculate how much we've actually read
                    let current_pos = self.reader.stream_position()?;
                    let bytes_read = current_pos
                        .checked_sub(chunk_start_pos)
                        .ok_or(Error::Overflow)?;
                    let remaining_in_chunk =
                        chunk_size.checked_sub(bytes_read).ok_or(Error::Overflow)?;

                    // Skip any remaining bytes in the chunk
                    if remaining_in_chunk > 0 {
                        self.skip_bytes(remaining_in_chunk)?;
                    }

                    // Process the chunk
                    f(chunk)?;
                }
                Ok(None) => { // Unknown chunk was skipped
                }
                Err(e) => {
                    // Try to recover by seeking to the end of the chunk
                    let target = chunk_start_pos
                        .checked_add(chunk_size)
                        .ok_or(Error::Overflow)?;
                    if let Ok(current) = self.reader.stream_position() {
                        if let Some(to_skip) = target.checked_sub(current) {
                            let _ = self.skip_bytes(to_skip);
                        }
                    }
                    // Return the original error after attempting to recover position
                    return Err(e);
                }
            }

            // Handle alignment for all chunks
            if needs_alignment {
                self.skip_bytes(1)?;
            }
        }

        Ok(())
    }

    fn parse_format_chunk(&mut self, chunk_size: usize) -> Result<Option<WaveChunk>> {
        if chunk_size < 16 {
            return Err(Error::Static(
                "Invalid format chunk size: must be at least 16 bytes",
            ));
        }

        let audio_format = self.read_u16_le()?;
        // CORRECTION #11: Reject non-PCM formats early.
        // WAVE_FORMAT_EXTENSIBLE (0xFFFE) is complex and not supported yet.
        if audio_format != 1 {
            return Err(Error::Static(
                "Unsupported audio format: only PCM (format code 1) is supported.",
            ));
        }

        let num_channels = self.read_u16_le()?;
        let sample_rate = self.read_u32_le()?;
        let byte_rate = self.read_u32_le()?;
        let block_align = self.read_u16_le()?;
        let bits_per_sample = self.read_u16_le()?;

        Ok(Some(WaveChunk::Format(FormatChunk {
            audio_format,
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
        })))
    }

    fn parse_data_chunk(&mut self, chunk_size: usize) -> Result<Option<WaveChunk>> {
        let data_position = self.reader.stream_position()?;

        // CORRECTION #1: Validate that data chunk does not exceed file length
        if data_position
            .checked_add(chunk_size as u64)
            .ok_or(Error::Overflow)?
            > self.file_len
        {
            return Err(Error::Static("Data chunk size exceeds file length."));
        }

        Ok(Some(WaveChunk::Data(DataChunk {
            length: chunk_size as u32,
            data_position,
        })))
    }

    fn parse_list_chunk(&mut self, chunk_size: usize) -> Result<Option<WaveChunk>> {
        const MAX_TAG_SIZE: usize = 1024 * 1024; // 1MB sanity limit per tag
        let list_type = self.read_exact::<4>()?;
        let mut tags = Vec::new();
        let mut remaining_size = chunk_size.checked_sub(4).ok_or(Error::Overflow)?;

        if &list_type == b"INFO" {
            while remaining_size >= 8 {
                // CORRECTION #9: Add a hard limit to the number of tags.
                if tags.len() >= MAX_TAGS {
                    break;
                };

                let tag_key_bytes = self.read_exact::<4>()?;
                let tag_size = self.read_u32_le()? as usize;

                remaining_size = remaining_size.checked_sub(8).ok_or(Error::Overflow)?;

                if tag_size > MAX_TAG_SIZE {
                    return Err(Error::Static("Tag size exceeds maximum allowed size"));
                }

                if tag_size > remaining_size {
                    return Err(Error::Static("Tag size exceeds remaining chunk size"));
                }

                let tag_value_bytes = self.read_bytes(tag_size)?;
                remaining_size = remaining_size
                    .checked_sub(tag_size)
                    .ok_or(Error::Overflow)?;

                if let Ok(key) = std::str::from_utf8(&tag_key_bytes) {
                    let value = String::from_utf8_lossy(&tag_value_bytes)
                        .trim_end_matches('\0')
                        .to_string();

                    tags.push(Tag {
                        tag_type: Some(TagType::from_bytes(&tag_key_bytes)),
                        key: key.to_string(),
                        value,
                    });
                }

                if tag_size % 2 != 0 {
                    self.skip_bytes(1)?;
                    remaining_size = remaining_size.checked_sub(1).ok_or(Error::Overflow)?;
                }
            }
        }

        Ok(Some(WaveChunk::List(ListChunk {
            _list_type: list_type,
            tags,
        })))
    }

    fn parse_fact_chunk(&mut self, _chunk_size: usize) -> Result<Option<WaveChunk>> {
        let num_samples = self.read_u32_le()?;
        Ok(Some(WaveChunk::Fact(FactChunk { num_samples })))
    }
}

#[derive(Debug)]
struct WavReaderOptions {
    codec_params: CodecParams,
    metadata: HashMap<String, String>,
    data_start: u64,
    data_end: u64,
}

/// WAV file reader.
#[derive(Debug)]
pub struct WavReader<R: Read + Seek> {
    cursor: u64,
    buf: BufReader<R>,
    options: WavReaderOptions,
}

impl<R: Read + Seek> WavReader<R> {
    const RIFF_HEADER: [u8; 4] = *b"RIFF";
    const WAVE_HEADER: [u8; 4] = *b"WAVE";
    const BUFFER_SIZE: usize = 1024 * 32;

    fn new(buf: BufReader<R>, options: WavReaderOptions) -> Self {
        let cursor = options.data_start;
        Self {
            buf,
            cursor,
            options,
        }
    }

    /// Attempts to create a new WavReader by parsing the WAV headers.
    pub fn try_new(mut file: R) -> Result<Self> {
        let file_len = file.seek(SeekFrom::End(0))?;
        file.seek(SeekFrom::Start(0))?;

        let mut buf = BufReader::with_capacity(Self::BUFFER_SIZE, file);

        let riff_header = {
            let mut riff_header = [0u8; 4];
            buf.read_exact(&mut riff_header)?;
            riff_header
        };
        if riff_header != Self::RIFF_HEADER {
            return Err(Error::Static("Invalid RIFF header"));
        }

        let chunk_size = {
            let mut size_bytes = [0u8; 4];
            buf.read_exact(&mut size_bytes)?;
            u32::from_le_bytes(size_bytes) as u64
        };

        if chunk_size > file_len.checked_sub(8).unwrap_or(0) {
            return Err(Error::Static("RIFF chunk size mismatch with file length"));
        }

        let wave_header = {
            let mut wave_header = [0u8; 4];
            buf.read_exact(&mut wave_header)?;
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

        let mut parser = ChunkParser::new(
            &mut buf,
            chunk_size.checked_sub(4).ok_or(Error::Overflow)?,
            file_len,
        );
        parser.parse_chunks(|chunk| match chunk {
            WaveChunk::Format(format) => Self::handle_format_chunk(&mut options, format),
            WaveChunk::Data(data) => Self::handle_data_chunk(&mut options, data),
            WaveChunk::Fact(fact) => Self::handle_fact_chunk(&mut options, fact),
            WaveChunk::List(list) => Self::handle_list_chunk(&mut options, list),
        })?;

        if options.codec_params.sample_format.is_none() {
            return Err(Error::Static("Missing required 'fmt ' chunk"));
        }

        Ok(Self::new(buf, options))
    }

    fn handle_format_chunk(options: &mut WavReaderOptions, chunk: FormatChunk) -> Result<()> {
        if chunk.block_align == 0 || chunk.num_channels == 0 || chunk.sample_rate == 0 {
            return Err(Error::Static("Format chunk contains invalid zero value(s)"));
        }

        options.codec_params.sample_rate = Some(chunk.sample_rate);
        options.codec_params.num_channels = chunk.num_channels;
        options.codec_params.bits_per_sample = Some(chunk.bits_per_sample);
        options.codec_params.block_align = Some(chunk.block_align);
        options.codec_params.audio_format = Some(chunk.audio_format);
        options.codec_params.byte_rate = Some(chunk.byte_rate);

        let sample_format = match chunk.bits_per_sample {
            8 => SampleFormat::Uint8,
            16 => SampleFormat::Int16,
            24 => SampleFormat::Int24,
            32 => SampleFormat::Int32,
            // CORRECTION #4: A full WAVE_FORMAT_EXTENSIBLE implementation is complex.
            // For now, we strictly check for known bits_per_sample values.
            // audio_format has already been checked in the parser.
            _ => return Err(Error::Static("Sample format not supported")),
        };
        options.codec_params.sample_format = Some(sample_format);

        Ok(())
    }

    fn handle_fact_chunk(options: &mut WavReaderOptions, chunk: FactChunk) -> Result<()> {
        // The 'fact' chunk often contains the total number of frames, which can be more reliable.
        options.codec_params.num_frames = Some(chunk.num_samples as u64);
        Ok(())
    }

    fn handle_data_chunk(options: &mut WavReaderOptions, chunk: DataChunk) -> Result<()> {
        options.data_start = chunk.data_position;
        options.data_end = chunk
            .data_position
            .checked_add(chunk.length as u64)
            .ok_or(Error::Overflow)?;

        // If 'fact' chunk was not present, calculate num_frames from data chunk size.
        if options.codec_params.num_frames.is_none() {
            if let Some(block_align) = options.codec_params.block_align {
                if block_align > 0 {
                    let num_frames = chunk.length as u64 / block_align as u64;
                    options.codec_params.num_frames = Some(num_frames);
                }
            }
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
        self.options.codec_params.num_frames.unwrap_or(0)
    }
    pub fn sample_rate(&self) -> u32 {
        self.options.codec_params.sample_rate.unwrap_or(0)
    }
    pub fn num_channels(&self) -> u16 {
        self.options.codec_params.num_channels
    }
    pub fn bits_per_sample(&self) -> u16 {
        self.options.codec_params.bits_per_sample.unwrap_or(0)
    }
    pub fn sample_format(&self) -> &SampleFormat {
        // This is safe because we check for sample_format.is_none() in try_new
        self.options.codec_params.sample_format.as_ref().unwrap()
    }
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.options.metadata
    }
    pub fn num_samples(&self) -> u64 {
        self.num_frames().saturating_mul(self.num_channels() as u64)
    }
    pub fn byte_rate(&self) -> u32 {
        self.options.codec_params.byte_rate.unwrap_or(0)
    }

    fn read_exact(&mut self, buffer: &mut [u8]) -> Result<()> {
        self.buf.read_exact(buffer)?;
        self.cursor = self
            .cursor
            .checked_add(buffer.len() as u64)
            .ok_or(Error::Overflow)?;
        Ok(())
    }
}

/// WAV file decoder.
pub struct WavDecoder<R: Read + Seek> {
    reader: WavReader<R>,
}

impl<R: Read + Seek> WavDecoder<R> {
    /// Creates a new WAV decoder from a `WavReader`.
    pub fn try_new(mut reader: WavReader<R>) -> Result<Self> {
        // The check for audio_format is now done in WavReader, so we can assume it's PCM.
        reader
            .buf
            .seek(SeekFrom::Start(reader.options.data_start))?;
        Ok(Self { reader })
    }

    /// Returns an iterator over decoded audio packets.
    pub fn packets(&mut self) -> Result<PacketsIterator<'_, R>> {
        PacketsIterator::new(&mut self.reader)
    }
}

struct PacketsIteratorParams {
    bytes_to_read: usize,
    total_samples: usize,
    bytes_per_sample: usize,
    is_data_available: bool,
}

/// Iterator over decoded audio samples.
pub struct PacketsIterator<'a, R: Read + Seek> {
    decoder: AudioDecoder,
    reader: &'a mut WavReader<R>,
    params: PacketsIteratorParams,
    rbuffer: AVec<u8, RuntimeAlign>,
    wbuffer: AVec<i32, RuntimeAlign>,
}

impl<'a, R: Read + Seek> PacketsIterator<'a, R> {
    fn new(reader: &'a mut WavReader<R>) -> Result<Self> {
        let decoder = AudioDecoder::new();
        let num_channels = reader.num_channels();
        let sample_format = reader.sample_format();
        let bytes_per_sample = sample_format.bytes_per_sample() as usize;

        // CORRECTION #10: Perform all calculations with overflow checks.
        let packet_len_bytes = (bytes_per_sample as usize)
            .checked_mul(num_channels as usize)
            .and_then(|v| v.checked_mul(PACK_SIZE))
            .ok_or(Error::Overflow)?;

        let num_samples = PACK_SIZE
            .checked_mul(num_channels as usize)
            .ok_or(Error::Overflow)?;

        let rbuffer = avec_rt!([32] | 0; packet_len_bytes);
        let wbuffer = avec_rt!([32] | 0; num_samples);

        let params = PacketsIteratorParams {
            bytes_per_sample,
            total_samples: 0,
            bytes_to_read: 0,
            is_data_available: true,
        };

        Ok(Self {
            reader,
            params,
            decoder,
            rbuffer,
            wbuffer,
        })
    }
}

impl<R: Read + Seek> FallibleStreamingIterator for PacketsIterator<'_, R> {
    type Item = [i32];
    type Error = Error;

    fn advance(&mut self) -> std::result::Result<(), Self::Error> {
        let params = &mut self.params;
        let cursor = self.reader.cursor;
        let data_end = self.reader.options.data_end;

        if cursor >= data_end {
            params.is_data_available = false;
            return Ok(());
        }

        let remaining = (data_end - cursor) as usize;
        params.bytes_to_read = remaining.min(self.rbuffer.len());

        if params.bytes_per_sample == 0 {
            return Err(Error::Static("Bytes per sample cannot be zero"));
        }
        params.total_samples = params.bytes_to_read / params.bytes_per_sample;

        if params.total_samples == 0 {
            params.is_data_available = false;
            return Ok(());
        }

        let rbuffer_slice = &mut self.rbuffer[..params.bytes_to_read];
        self.reader.read_exact(rbuffer_slice)?;

        let wbuffer_slice = &mut self.wbuffer[..params.total_samples];

        let format = *self.reader.sample_format();
        use SampleFormat as SF;

        match format {
            SF::Uint8 => self.decoder.decode_uint8(rbuffer_slice, wbuffer_slice),
            SF::Int16 => self.decoder.decode_int16(rbuffer_slice, wbuffer_slice),
            SF::Int24 => self.decoder.decode_int24(rbuffer_slice, wbuffer_slice),
            SF::Int32 => self.decoder.decode_int32(rbuffer_slice, wbuffer_slice),
        }

        Ok(())
    }

    fn get(&self) -> Option<&Self::Item> {
        if !self.params.is_data_available {
            return None;
        }

        Some(&self.wbuffer[..self.params.total_samples])
    }
}

/// Trait defining the decoding operations.
#[enum_dispatch]
trait Decoder {
    fn decode_uint8(&self, rbuffer: &[u8], wbuffer: &mut [i32]);
    fn decode_int16(&self, rbuffer: &[u8], wbuffer: &mut [i32]);
    fn decode_int24(&self, rbuffer: &[u8], wbuffer: &mut [i32]);
    fn decode_int32(&self, rbuffer: &[u8], wbuffer: &mut [i32]);
}

/// Enum that will dispatch to the correct decoder implementation at runtime.
#[enum_dispatch(Decoder)]
enum AudioDecoder {
    Avx2Decoder,
    Sse42Decoder,
    ScalarDecoder,
}

impl AudioDecoder {
    /// Creates a new AudioDecoder by detecting CPU features at runtime.
    fn new() -> Self {
        if is_x86_feature_detected!("avx2") {
            return Avx2Decoder.into();
        }
        if is_x86_feature_detected!("sse4.1") {
            return Sse42Decoder.into();
        }
        ScalarDecoder.into()
    }
}

// CORRECTION #2: Add cfg attributes for compile-time safety
#[cfg(target_arch = "x86_64")]
#[derive(Default)]
struct Avx2Decoder;

#[cfg(target_arch = "x86_64")]
impl Decoder for Avx2Decoder {
    fn decode_uint8(&self, rbuffer: &[u8], wbuffer: &mut [i32]) {
        debug_assert!(is_x86_feature_detected!("avx2"));
        unsafe { decode_uint8_avx2(rbuffer, wbuffer, wbuffer.len()) };
    }
    fn decode_int16(&self, rbuffer: &[u8], wbuffer: &mut [i32]) {
        debug_assert!(is_x86_feature_detected!("avx2"));
        unsafe { decode_int16_avx2(rbuffer, wbuffer, wbuffer.len()) };
    }
    fn decode_int24(&self, rbuffer: &[u8], wbuffer: &mut [i32]) {
        debug_assert!(is_x86_feature_detected!("avx2"));
        unsafe { decode_int24_avx2(rbuffer, wbuffer, wbuffer.len()) };
    }
    fn decode_int32(&self, rbuffer: &[u8], wbuffer: &mut [i32]) {
        debug_assert!(is_x86_feature_detected!("avx2"));
        unsafe { decode_int32_avx2(rbuffer, wbuffer, wbuffer.len()) };
    }
}

#[cfg(target_arch = "x86_64")]
#[derive(Default)]
struct Sse42Decoder;

#[cfg(target_arch = "x86_64")]
impl Decoder for Sse42Decoder {
    fn decode_uint8(&self, rbuffer: &[u8], wbuffer: &mut [i32]) {
        debug_assert!(is_x86_feature_detected!("sse4.1"));
        unsafe { decode_uint8_sse42(rbuffer, wbuffer, wbuffer.len()) };
    }
    fn decode_int16(&self, rbuffer: &[u8], wbuffer: &mut [i32]) {
        debug_assert!(is_x86_feature_detected!("sse4.1"));
        unsafe { decode_int16_sse42(rbuffer, wbuffer, wbuffer.len()) };
    }
    fn decode_int24(&self, rbuffer: &[u8], wbuffer: &mut [i32]) {
        debug_assert!(is_x86_feature_detected!("sse4.1"));
        unsafe { decode_int24_sse42(rbuffer, wbuffer, wbuffer.len()) };
    }
    fn decode_int32(&self, rbuffer: &[u8], wbuffer: &mut [i32]) {
        debug_assert!(is_x86_feature_detected!("sse4.1"));
        unsafe { decode_int32_sse42(rbuffer, wbuffer, wbuffer.len()) };
    }
}

#[derive(Default)]
struct ScalarDecoder;
impl Decoder for ScalarDecoder {
    fn decode_uint8(&self, rbuffer: &[u8], wbuffer: &mut [i32]) {
        for (i, sample) in rbuffer.iter().enumerate().take(wbuffer.len()) {
            wbuffer[i] = u8_to_i32(*sample);
        }
    }
    fn decode_int16(&self, rbuffer: &[u8], wbuffer: &mut [i32]) {
        // CORRECTION #15: Use bytemuck for safe, efficient casting
        let samples = bytemuck::cast_slice::<u8, i16>(rbuffer);
        for (i, &sample) in samples.iter().enumerate().take(wbuffer.len()) {
            wbuffer[i] = sample as i32;
        }
    }
    fn decode_int24(&self, rbuffer: &[u8], wbuffer: &mut [i32]) {
        let chunks = rbuffer.chunks_exact(3);
        for (i, chunk) in chunks.enumerate().take(wbuffer.len()) {
            wbuffer[i] = i24_to_i32(chunk.try_into().unwrap());
        }
    }
    fn decode_int32(&self, rbuffer: &[u8], wbuffer: &mut [i32]) {
        // CORRECTION #15: Use bytemuck for safe, efficient casting
        let samples = bytemuck::cast_slice::<u8, i32>(rbuffer);
        for (i, &sample) in samples.iter().enumerate().take(wbuffer.len()) {
            wbuffer[i] = sample;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decode_uint8_avx2(rbuffer: &[u8], wbuffer: &mut [i32], total_samples: usize) {
    let mut i = 0;
    let mut out_ptr = wbuffer.as_mut_ptr();

    while i + 16 <= total_samples {
        let chunk_u8 = _mm_loadu_si128(rbuffer.as_ptr().add(i) as *const __m128i);
        let chunk_i16 = _mm256_cvtepu8_epi16(chunk_u8);
        let offset = _mm256_set1_epi16(128);
        let adjusted_i16 = _mm256_sub_epi16(chunk_i16, offset);

        let low_8_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(adjusted_i16));
        let high_8_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(adjusted_i16));

        _mm256_storeu_si256(out_ptr as *mut __m256i, low_8_i32);
        _mm256_storeu_si256(out_ptr.add(8) as *mut __m256i, high_8_i32);

        out_ptr = out_ptr.add(16);
        i += 16;
    }

    if i < total_samples {
        ScalarDecoder.decode_uint8(&rbuffer[i..], &mut wbuffer[i..]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decode_int16_avx2(rbuffer: &[u8], wbuffer: &mut [i32], total_samples: usize) {
    let mut i = 0;
    let mut out_ptr = wbuffer.as_mut_ptr();

    while i + 16 <= total_samples {
        let in_ptr = rbuffer.as_ptr().add(i * 2);
        let chunk = _mm256_loadu_si256(in_ptr as *const __m256i);
        let low_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(chunk));
        let high_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(chunk));

        _mm256_storeu_si256(out_ptr as *mut __m256i, low_i32);
        _mm256_storeu_si256(out_ptr.add(8) as *mut __m256i, high_i32);

        out_ptr = out_ptr.add(16);
        i += 16;
    }
    if i < total_samples {
        ScalarDecoder.decode_int16(&rbuffer[i * 2..], &mut wbuffer[i..]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decode_int24_avx2(rbuffer: &[u8], wbuffer: &mut [i32], _total_samples: usize) {
    // ------------------------------------------------------------------
    // Implementación temporal: delegamos al decodificador escalar
    // hasta que pulamos una versión AVX2 correcta y verificada.
    // Esto garantiza resultados idénticos y corrige el fallo del test
    // `decoder_equivalence` (mismatch en Int24).
    // ------------------------------------------------------------------
    ScalarDecoder.decode_int24(rbuffer, wbuffer);

    // NOTA:
    // Si más adelante deseas re-activar la ruta SIMD, reemplaza el cuerpo
    // por la rutina vectorizada definitiva y vuelve a ejecutar los tests.
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decode_int32_avx2(rbuffer: &[u8], wbuffer: &mut [i32], total_samples: usize) {
    let mut i = 0;
    let mut out_ptr = wbuffer.as_mut_ptr();

    while i + 8 <= total_samples {
        let in_ptr = rbuffer.as_ptr().add(i * 4);
        let chunk = _mm256_loadu_si256(in_ptr as *const __m256i);
        _mm256_storeu_si256(out_ptr as *mut __m256i, chunk);
        out_ptr = out_ptr.add(8);
        i += 8;
    }
    if i < total_samples {
        ScalarDecoder.decode_int32(&rbuffer[i * 4..], &mut wbuffer[i..]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn decode_uint8_sse42(rbuffer: &[u8], wbuffer: &mut [i32], total_samples: usize) {
    let mut i = 0;
    let mut out_ptr = wbuffer.as_mut_ptr();

    while i + 8 <= total_samples {
        let chunk_u8 = _mm_loadl_epi64(rbuffer.as_ptr().add(i) as *const __m128i);
        let chunk_i16 = _mm_cvtepu8_epi16(chunk_u8);
        let offset = _mm_set1_epi16(128);
        let adjusted = _mm_sub_epi16(chunk_i16, offset);

        let low_i32 = _mm_cvtepi16_epi32(adjusted);
        let high_i32 = _mm_cvtepi16_epi32(_mm_srli_si128(adjusted, 8));

        _mm_storeu_si128(out_ptr as *mut __m128i, low_i32);
        _mm_storeu_si128(out_ptr.add(4) as *mut __m128i, high_i32);

        out_ptr = out_ptr.add(8);
        i += 8;
    }
    if i < total_samples {
        ScalarDecoder.decode_uint8(&rbuffer[i..], &mut wbuffer[i..]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn decode_int16_sse42(rbuffer: &[u8], wbuffer: &mut [i32], total_samples: usize) {
    let mut i = 0;
    let mut out_ptr = wbuffer.as_mut_ptr();

    while i + 8 <= total_samples {
        let in_ptr = rbuffer.as_ptr().add(i * 2);
        let chunk = _mm_loadu_si128(in_ptr as *const __m128i);

        let low_i32 = _mm_cvtepi16_epi32(chunk);
        _mm_storeu_si128(out_ptr as *mut __m128i, low_i32);

        let high_part = _mm_srli_si128(chunk, 8);
        let high_i32 = _mm_cvtepi16_epi32(high_part);
        _mm_storeu_si128(out_ptr.add(4) as *mut __m128i, high_i32);

        out_ptr = out_ptr.add(8);
        i += 8;
    }
    if i < total_samples {
        ScalarDecoder.decode_int16(&rbuffer[i * 2..], &mut wbuffer[i..]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn decode_int24_sse42(rbuffer: &[u8], wbuffer: &mut [i32], total_samples: usize) {
    let mut i = 0;
    let mut out_ptr = wbuffer.as_mut_ptr();
    let shuffle_mask = _mm_set_epi8(-1, 11, 10, 9, -1, 8, 7, 6, -1, 5, 4, 3, -1, 2, 1, 0);

    while i + 4 <= total_samples {
        let ptr = rbuffer.as_ptr().add(i * 3);
        let data_chunk = _mm_loadu_si128(ptr as *const __m128i);

        let expanded = _mm_shuffle_epi8(data_chunk, shuffle_mask);
        let shifted_left = _mm_slli_epi32(expanded, 8);
        let sign_extended = _mm_srai_epi32(shifted_left, 8);

        _mm_storeu_si128(out_ptr as *mut __m128i, sign_extended);

        out_ptr = out_ptr.add(4);
        i += 4;
    }
    if i < total_samples {
        ScalarDecoder.decode_int24(&rbuffer[i * 3..], &mut wbuffer[i..]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn decode_int32_sse42(rbuffer: &[u8], wbuffer: &mut [i32], total_samples: usize) {
    let mut i = 0;
    let mut out_ptr = wbuffer.as_mut_ptr();

    while i + 4 <= total_samples {
        let in_ptr = rbuffer.as_ptr().add(i * 4);
        let chunk = _mm_loadu_si128(in_ptr as *const __m128i);
        _mm_storeu_si128(out_ptr as *mut __m128i, chunk);
        out_ptr = out_ptr.add(4);
        i += 4;
    }
    if i < total_samples {
        ScalarDecoder.decode_int32(&rbuffer[i * 4..], &mut wbuffer[i..]);
    }
}

// CORRECTION #14: Helper functions are simplified and can be const
#[inline(always)]
pub const fn u8_to_i32(byte: u8) -> i32 {
    (byte as i32) - 128
}

#[inline(always)]
pub const fn i16_to_i32(bytes: [u8; 2]) -> i32 {
    i16::from_le_bytes(bytes) as i32
}

#[inline(always)]
pub const fn i24_to_i32(bytes: [u8; 3]) -> i32 {
    let val = (bytes[0] as i32) | ((bytes[1] as i32) << 8) | ((bytes[2] as i32) << 16);
    // Sign-extend the 24-bit value
    (val << 8) >> 8
}

#[inline(always)]
pub const fn i32_to_i32(bytes: [u8; 4]) -> i32 {
    i32::from_le_bytes(bytes)
}

// ───────────────────────────────────────────────────────────────────────────
// TESTS (todo en el mismo lib.rs)
// ───────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::io::Cursor;

    // ---------- util: helpers compartidos ----------
    mod util {
        use super::*;

        /// RNG reproducible para todas las pruebas.
        pub fn rng() -> StdRng {
            StdRng::seed_from_u64(0xD1CE_BEEF)
        }

        /// Encabezado WAV + sección `data` opcional.
        pub fn wav_header_with_data(sr: u32, ch: u16, bps: u16, data_len: u32) -> Vec<u8> {
            let byte_rate = sr
                .saturating_mul((bps / 8) as u32)
                .saturating_mul(ch as u32);
            let block_align = bps / 8 * ch;
            let total = 36 + data_len;

            let mut v = Vec::with_capacity(total as usize + 8);
            v.extend_from_slice(b"RIFF");
            v.extend_from_slice(&total.to_le_bytes());
            v.extend_from_slice(b"WAVE");

            v.extend_from_slice(b"fmt ");
            v.extend_from_slice(&16u32.to_le_bytes());
            v.extend_from_slice(&1u16.to_le_bytes()); // PCM
            v.extend_from_slice(&ch.to_le_bytes());
            v.extend_from_slice(&sr.to_le_bytes());
            v.extend_from_slice(&byte_rate.to_le_bytes());
            v.extend_from_slice(&block_align.to_le_bytes());
            v.extend_from_slice(&bps.to_le_bytes());

            v.extend_from_slice(b"data");
            v.extend_from_slice(&data_len.to_le_bytes());
            v
        }
        pub fn minimal_header(sr: u32, ch: u16, bps: u16) -> Vec<u8> {
            wav_header_with_data(sr, ch, bps, 0)
        }
    }

    // ---------- pruebas de parseo ----------
    #[test]
    fn invalid_headers() {
        use util::*;
        // RIFF incorrecto
        let mut bad = minimal_header(44_100, 1, 16);
        bad[..4].copy_from_slice(b"XXXX");
        assert!(matches!(
            WavReader::try_new(Cursor::new(bad)),
            Err(Error::Static("Invalid RIFF header"))
        ));

        // WAVE incorrecto
        let mut bad = minimal_header(48_000, 1, 16);
        bad[8..12].copy_from_slice(b"XXXX");
        assert!(matches!(
            WavReader::try_new(Cursor::new(bad)),
            Err(Error::Static("Invalid WAVE header"))
        ));
    }

    #[test]
    fn list_chunk_metadata() {
        use util::*;
        let mut v = minimal_header(48_000, 1, 16);
        let list = [
            b"LIST".as_ref(),     // &[u8]
            &18u32.to_le_bytes(), // &[u8; 4]  → &[u8]
            b"INFO".as_ref(),
            b"IART".as_ref(),
            &5u32.to_le_bytes(),
            b"Test\0".as_ref(), // 5 bytes
            &[0u8][..],         // 1 byte
        ]
        .concat(); // Vec<u8>
        v.extend(list);
        let sz = (v.len() - 8) as u32;
        v[4..8].copy_from_slice(&sz.to_le_bytes());

        let rdr = WavReader::try_new(Cursor::new(v)).unwrap();
        assert_eq!(rdr.metadata().get("IART"), Some(&"Test".to_string()));
    }

    // ---------- equivalencia de decodificadores ----------
    fn assert_equiv<D: Decoder>(dec: D, fmt: SampleFormat) {
        let mut rng = util::rng();
        let n = 64usize;
        let src: Vec<u8> = (0..n * fmt.bytes_per_sample() as usize)
            .map(|_| rng.random())
            .collect();
        let mut ref_buf = vec![0i32; n];
        let mut test_buf = vec![0i32; n];

        match fmt {
            SampleFormat::Uint8 => {
                ScalarDecoder.decode_uint8(&src, &mut ref_buf);
                dec.decode_uint8(&src, &mut test_buf);
            }
            SampleFormat::Int16 => {
                ScalarDecoder.decode_int16(&src, &mut ref_buf);
                dec.decode_int16(&src, &mut test_buf);
            }
            SampleFormat::Int24 => {
                ScalarDecoder.decode_int24(&src, &mut ref_buf);
                dec.decode_int24(&src, &mut test_buf);
            }
            SampleFormat::Int32 => {
                ScalarDecoder.decode_int32(&src, &mut ref_buf);
                dec.decode_int32(&src, &mut test_buf);
            }
        }
        assert_eq!(ref_buf, test_buf, "Mismatch in {:?}", fmt);
    }

    #[test]
    fn decoder_equivalence() {
        for fmt in [
            SampleFormat::Uint8,
            SampleFormat::Int16,
            SampleFormat::Int24,
            SampleFormat::Int32,
        ] {
            assert_equiv(ScalarDecoder, fmt);
            if is_x86_feature_detected!("avx2") {
                assert_equiv(Avx2Decoder, fmt);
            }
            if is_x86_feature_detected!("sse4.1") {
                assert_equiv(Sse42Decoder, fmt);
            }
        }
    }

    // ---------- lógica del iterador ----------
    #[test]
    fn iterator_empty_data() {
        use util::*;
        let hdr = minimal_header(16_000, 1, 8);
        let mut binding =
            WavDecoder::try_new(WavReader::try_new(Cursor::new(hdr)).unwrap()).unwrap();
        let mut pk = binding.packets().unwrap();
        assert!(pk.next().unwrap().is_none());
    }

    // ---------- módulo de fuzz y edge cases ----------
    mod fuzz {
        use super::util::*;
        use super::*;

        #[test]
        fn random_headers_no_panic() {
            let mut rng = rng();
            for _ in 0..1_000 {
                let mut hdr = vec![0u8; 64];
                rng.fill(&mut hdr[..]);
                let _ = WavReader::try_new(Cursor::new(hdr));
            }
        }

        #[test]
        fn raw_random_bytes() {
            let mut rng = rng();
            for _ in 0..500 {
                let len = rng.random_range(16..=1024);
                let bytes: Vec<u8> = (0..len).map(|_| rng.random()).collect();
                let _ = WavReader::try_new(Cursor::new(bytes));
            }
        }
    }
}
