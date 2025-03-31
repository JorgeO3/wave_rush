#![cfg(target_arch = "x86_64")] // Indicate that SIMD parts target x86_64

use std::arch::x86_64::*;
use std::collections::HashMap;
use std::fmt::Debug;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::marker::PhantomData; // Needed for Decoder trait lifetime variance

// Use aligned_vec crate for guaranteed alignment
use aligned_vec::{avec_rt, AVec, RuntimeAlign};

// Use fallible_streaming_iterator for efficient iteration
pub use fallible_streaming_iterator::FallibleStreamingIterator;

/// Number of samples (per channel) per processing packet.
/// Chosen to be reasonably large for SIMD efficiency.
pub const PACK_SIZE: usize = 1024; // Reduced from 8k for potentially better cache behavior

// Compile-time validations for PACK_SIZE based on SIMD register sizes
// AVX2 works with 32 bytes (256 bits) -> 8xi32, 16xi16, 32xu8
// SSE works with 16 bytes (128 bits) -> 4xi32, 8xi16, 16xu8
const _: () = assert!(
    PACK_SIZE % 32 == 0,
    "PACK_SIZE must be multiple of 32 for AVX2 uint8"
);
const _: () = assert!(
    PACK_SIZE % 16 == 0,
    "PACK_SIZE must be multiple of 16 for AVX2 int16"
);
const _: () = assert!(
    PACK_SIZE % 8 == 0,
    "PACK_SIZE must be multiple of 8 for AVX2/SSE int32"
);
const _: () = assert!(
    PACK_SIZE % 16 == 0,
    "PACK_SIZE must be multiple of 16 for SSE uint8"
);
const _: () = assert!(
    PACK_SIZE % 8 == 0,
    "PACK_SIZE must be multiple of 8 for SSE int16"
);

/// Specialized Result type for this crate's operations.
pub type Result<T> = std::result::Result<T, Error>;

/// General errors for this crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Invalid WAV file: {0}")]
    Static(&'static str),
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("UTF-8 decoding error in metadata: {0}")]
    Utf8(#[from] std::str::Utf8Error),
    #[error("Unsupported WAV format: {0}")]
    Unsupported(&'static str),
}

/// Represents different PCM sample formats found in WAV files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    /// Unsigned 8-bit integer samples.
    Uint8,
    /// Signed 16-bit integer samples (little-endian).
    Int16,
    /// Signed 24-bit integer samples (little-endian, packed).
    Int24,
    /// Signed 32-bit integer samples (little-endian).
    Int32,
}

impl SampleFormat {
    /// Returns the number of bytes per sample for this format.
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

/// Holds the parsed codec parameters from the WAV file's 'fmt ' chunk.
#[derive(Debug, Default, Clone)]
struct CodecParams {
    sample_rate: Option<u32>,
    num_frames: Option<u64>,
    sample_format: Option<SampleFormat>,
    bits_per_sample: Option<u16>,
    num_channels: u16,
    block_align: Option<u16>,  // Bytes per multi-channel sample frame
    audio_format: Option<u16>, // e.g., 1 for PCM
    byte_rate: Option<u32>,    // sample_rate * block_align
}

/// A Tag holds a key-value pair of metadata from an INFO chunk.
#[derive(Clone, Debug)]
pub struct Tag {
    pub tag_type: Option<TagType>,
    pub key: String, // The original 4-char key
    pub value: String,
}

/// Known metadata tag types based on common INFO chunk keys.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TagType {
    Rating,
    Comment,
    OriginalDate,
    Genre,
    Artist,
    Copyright,
    Date, // Creation date
    EncodedBy,
    Engineer,
    TrackTotal, // Frames or total tracks in album (context dependent)
    Language,
    Composer,
    TrackTitle,
    Album,
    Producer,
    TrackNumber,
    Encoder,     // Software used for encoding
    MediaFormat, // Source media (e.g., "CD", "DAT")
    Writer,      // Often Lyricist or Text writer
    Label,       // Original Publisher/Label
    Version,     // Version of the title/track
    Unknown,     // Key not recognized
}

impl TagType {
    /// Converts a 4-byte LIST INFO key into a known `TagType`.
    fn from_bytes(key: &[u8; 4]) -> Self {
        match key {
            // Common variations observed in the wild
            b"IART" => TagType::Artist,                // Artist
            b"ICMT" | b"COMM" => TagType::Comment,     // Comment
            b"ICOP" => TagType::Copyright,             // Copyright
            b"ICRD" | b"IDIT" => TagType::Date,        // Creation Date or Digitization Date
            b"IENG" => TagType::Engineer,              // Engineer
            b"IGNR" | b"GENR" => TagType::Genre,       // Genre
            b"INAM" | b"TITL" => TagType::TrackTitle,  // Title / Name of subject
            b"IPRD" => TagType::Album,                 // Product / Album Title
            b"IPRT" | b"TRCK" => TagType::TrackNumber, // Part / Track Number
            b"ISFT" => TagType::Encoder,               // Software / Encoder
            b"ISRC" => TagType::Unknown, // ISRC (International Standard Recording Code) - Often just Unknown
            b"ISBJ" => TagType::Unknown, // Subject - Often just Unknown
            b"ITCH" => TagType::EncodedBy, // Technician
            // Less common but possible
            b"AGES" => TagType::Rating,
            b"ILNG" | b"LANG" => TagType::Language,
            b"IMUS" => TagType::Composer,
            b"IPRO" => TagType::Producer,
            b"ISRF" => TagType::MediaFormat,
            b"IWRI" => TagType::Writer,
            b"TORG" => TagType::Label, // Original Organization / Label
            b"TVER" => TagType::Version,
            // Date/Time specific
            b"DTIM" => TagType::OriginalDate, // Digitization Time
            b"YEAR" => TagType::Date,         // Year (often redundant with ICRD)
            // Frame related (might be in 'fact' or context)
            b"IFRM" => TagType::TrackTotal,
            _ => TagType::Unknown,
        }
    }
}

/// Represents the content of the "fmt " chunk in a WAV file.
struct FormatChunk {
    audio_format: u16,
    num_channels: u16,
    sample_rate: u32,
    byte_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
}

/// Represents the content of a "LIST" chunk (specifically focusing on INFO tags).
struct ListChunk {
    _list_type: [u8; 4], // e.g., "INFO"
    tags: Vec<Tag>,
}

/// Represents the content of the "fact" chunk (often contains total sample frames).
struct FactChunk {
    num_sample_frames: u32, // Number of sample frames across all channels
}

/// Represents the location and size of the "data" chunk.
struct DataChunk {
    size: u32,           // Size of the audio data in bytes
    data_start_pos: u64, // File offset where the actual audio data begins
}

/// Enum wrapping the different parsed WAV chunk types this library handles.
enum ParsedWaveChunk {
    Format(FormatChunk),
    List(ListChunk),
    Fact(FactChunk),
    Data(DataChunk),
}

/// Internal enum to identify chunk types quickly during parsing.
enum ChunkId {
    Format,
    List,
    Fact,
    Data,
    Unknown,
}

impl ChunkId {
    /// Converts a 4-byte chunk ID into the corresponding `ChunkId`.
    #[inline(always)]
    fn from_bytes(id: &[u8; 4]) -> Self {
        match id {
            b"fmt " => ChunkId::Format,
            b"LIST" => ChunkId::List,
            b"fact" => ChunkId::Fact,
            b"data" => ChunkId::Data,
            _ => ChunkId::Unknown,
        }
    }
}

/// A parser that iterates over and interprets RIFF chunks within a WAV file.
/// It reads from a `BufReader` and keeps track of its position within the RIFF chunk boundary.
struct ChunkParser<'a, R: Read + Seek + Debug> {
    reader: &'a mut BufReader<R>,
    cursor: u64,         // Use u64 to match SeekFrom
    riff_chunk_end: u64, // The end offset of the main RIFF chunk
}

impl<'a, R: Read + Seek + Debug> ChunkParser<'a, R> {
    /// Creates a new chunk parser.
    /// `reader`: The buffered reader for the file.
    /// `riff_chunk_size`: The size reported in the RIFF header (file size - 8).
    /// `start_offset`: The file offset where the RIFF chunk content begins (after "RIFF" and size).
    pub fn new(reader: &'a mut BufReader<R>, riff_chunk_size: u32, start_offset: u64) -> Self {
        Self {
            reader,
            cursor: start_offset,
            riff_chunk_end: start_offset + riff_chunk_size as u64,
        }
    }

    /// Aligns the read cursor to the next even byte boundary if necessary,
    /// as required by the RIFF specification after each chunk.
    #[inline]
    fn align_cursor(&mut self) -> Result<()> {
        if self.cursor % 2 != 0 {
            // Read and discard one padding byte
            let mut padding = [0u8; 1];
            if self.reader.read(&mut padding)? == 1 {
                self.cursor += 1;
            } else {
                // If we can't read the padding byte, we might be exactly at EOF
                // which is okay if the chunk ended exactly at the file end.
                // If not, it's an unexpected EOF.
                if self.cursor < self.riff_chunk_end {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "Failed to read padding byte before end of RIFF chunk",
                    )
                    .into());
                }
            }
        }
        Ok(())
    }

    /// Skips `n` bytes in the reader, updating the cursor.
    fn skip_bytes(&mut self, n: u64) -> Result<()> {
        // Seek is often faster than reading into a buffer
        self.reader.seek(SeekFrom::Current(n as i64))?;
        self.cursor += n;
        Ok(())
    }

    /// Reads exactly `N` bytes into a fixed-size array.
    #[inline]
    fn read_exact<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut buf = [0u8; N];
        self.reader.read_exact(&mut buf)?;
        self.cursor += N as u64;
        Ok(buf)
    }

    /// Reads a little-endian u16.
    #[inline]
    fn read_u16_le(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.read_exact::<2>()?))
    }

    /// Reads a little-endian u32.
    #[inline]
    fn read_u32_le(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read_exact::<4>()?))
    }

    /// Reads `n` bytes into a `Vec<u8>`.
    fn read_vec(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0; n];
        self.reader.read_exact(&mut buf)?;
        self.cursor += n as u64;
        Ok(buf)
    }

    /// Parses all chunks within the RIFF boundary, calling `f` for each recognized chunk.
    /// Skips unknown chunks.
    pub fn parse_chunks<F>(&mut self, mut f: F) -> Result<()>
    where
        F: FnMut(ParsedWaveChunk) -> Result<()>,
    {
        // Need at least 8 bytes for ID + size
        while self.cursor + 8 <= self.riff_chunk_end {
            let chunk_id_bytes = self.read_exact::<4>()?;
            let chunk_size = self.read_u32_le()?;
            let chunk_id = ChunkId::from_bytes(&chunk_id_bytes);

            let chunk_end_cursor = self.cursor + chunk_size as u64;

            // Check for overflow and bounds
            if chunk_end_cursor < self.cursor || chunk_end_cursor > self.riff_chunk_end {
                return Err(Error::Static(
                    "Chunk size exceeds RIFF boundary or causes overflow",
                ));
            }

            let current_chunk_start_cursor = self.cursor;

            let result = match chunk_id {
                ChunkId::Format => self.parse_format_chunk(chunk_size),
                ChunkId::Data => self.parse_data_chunk(chunk_size),
                ChunkId::List => self.parse_list_chunk(chunk_size),
                ChunkId::Fact => self.parse_fact_chunk(chunk_size),
                ChunkId::Unknown => {
                    // Skip unknown chunk by seeking past its content
                    self.skip_bytes(chunk_size as u64)?;
                    Ok(None) // Indicate no parsed chunk to process
                }
            };

            match result {
                Ok(Some(parsed_chunk)) => {
                    f(parsed_chunk)?;
                }
                Ok(None) => {
                    // Unknown chunk was skipped, continue
                }
                Err(e) => {
                    // Propagate parsing errors
                    return Err(e);
                }
            }

            // Ensure cursor is where it should be after processing/skipping the chunk content
            if self.cursor != chunk_end_cursor {
                // This indicates a logic error in one of the parse_* methods or skip_bytes
                eprintln!("Warning: Cursor mismatch after chunk {:?} (expected {}, got {}). Attempting to correct.",
                           std::str::from_utf8(&chunk_id_bytes).unwrap_or("????"),
                           chunk_end_cursor, self.cursor);
                // Try to recover by seeking to the expected end
                let needed_seek = chunk_end_cursor as i64 - self.cursor as i64;
                self.reader.seek(SeekFrom::Current(needed_seek))?;
                self.cursor = chunk_end_cursor;
            }

            // Align cursor to an even boundary for the next chunk, only if necessary
            self.align_cursor()?;
        }

        // Ensure we didn't stop mid-chunk header if the loop terminated early
        if self.cursor < self.riff_chunk_end {
            // It's possible the last chunk was slightly malformed or the RIFF size was wrong.
            // We could issue a warning or error here. Let's warn for now.
            eprintln!(
                "Warning: Reached end of parsing with {} bytes remaining in RIFF chunk.",
                self.riff_chunk_end - self.cursor
            );
        }

        Ok(())
    }

    /// Parses the 'fmt ' chunk content.
    fn parse_format_chunk(&mut self, chunk_size: u32) -> Result<Option<ParsedWaveChunk>> {
        // Basic PCM format chunk requires at least 16 bytes
        if chunk_size < 16 {
            return Err(Error::Static("Invalid 'fmt ' chunk size (< 16 bytes)"));
        }

        let audio_format = self.read_u16_le()?;
        let num_channels = self.read_u16_le()?;
        let sample_rate = self.read_u32_le()?;
        let byte_rate = self.read_u32_le()?;
        let block_align = self.read_u16_le()?;
        let bits_per_sample = self.read_u16_le()?;

        // The 'fmt ' chunk can have extension data after the basic 16 bytes.
        // We need to skip it if present.
        let remaining_bytes = chunk_size.saturating_sub(16);
        if remaining_bytes > 0 {
            self.skip_bytes(remaining_bytes as u64)?;
        }

        Ok(Some(ParsedWaveChunk::Format(FormatChunk {
            audio_format,
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
        })))
    }

    /// Parses the 'data' chunk header to find the data location and size.
    /// Does *not* read the data itself, only skips past it.
    fn parse_data_chunk(&mut self, chunk_size: u32) -> Result<Option<ParsedWaveChunk>> {
        let data_start_pos = self.cursor; // Current position is start of data
        self.skip_bytes(chunk_size as u64)?; // Skip the entire data section
        Ok(Some(ParsedWaveChunk::Data(DataChunk {
            size: chunk_size,
            data_start_pos,
        })))
    }

    /// Parses the 'LIST' chunk, currently only handling 'INFO' sub-type for metadata.
    fn parse_list_chunk(&mut self, chunk_size: u32) -> Result<Option<ParsedWaveChunk>> {
        if chunk_size < 4 {
            return Err(Error::Static("Invalid 'LIST' chunk size (< 4 bytes)"));
        }

        let list_type = self.read_exact::<4>()?;
        let mut tags = Vec::new();
        let content_size = chunk_size - 4; // Size of the content after "INFO" etc.

        if &list_type == b"INFO" {
            // Parse INFO tags within the bounds of this LIST chunk
            self.parse_info_tags(content_size, &mut tags)?;
        } else {
            // Skip content of unknown/unsupported LIST types (e.g., 'adtl')
            self.skip_bytes(content_size as u64)?;
        }

        Ok(Some(ParsedWaveChunk::List(ListChunk {
            _list_type: list_type,
            tags,
        })))
    }

    /// Parses key-value metadata tags within an 'INFO' list chunk.
    fn parse_info_tags(&mut self, mut remaining_info_size: u32, tags: &mut Vec<Tag>) -> Result<()> {
        while remaining_info_size >= 8 {
            // Need at least 4 bytes for key, 4 bytes for size
            let tag_key_bytes = self.read_exact::<4>()?;
            let tag_value_size = self.read_u32_le()?;
            remaining_info_size = remaining_info_size.saturating_sub(8);

            if tag_value_size > remaining_info_size {
                return Err(Error::Static(
                    "INFO tag size exceeds remaining LIST chunk size",
                ));
            }

            let value_bytes = self.read_vec(tag_value_size as usize)?;
            remaining_info_size = remaining_info_size.saturating_sub(tag_value_size);

            // Values are null-terminated C strings. Trim null bytes.
            // Find the first null byte, or use the full length if none.
            let actual_len = value_bytes
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(value_bytes.len());
            // Lossy conversion is acceptable for metadata.
            let value_str = String::from_utf8_lossy(&value_bytes[..actual_len]).to_string();

            // Keys are typically ASCII, but handle potential errors.
            let key_str = std::str::from_utf8(&tag_key_bytes)?.to_string();

            let tag_type = TagType::from_bytes(&tag_key_bytes);
            tags.push(Tag {
                tag_type: Some(tag_type),
                key: key_str,
                value: value_str,
            });

            // Tag data is also padded to an even size.
            if tag_value_size % 2 != 0 {
                if remaining_info_size >= 1 {
                    self.skip_bytes(1)?; // Skip padding byte
                    remaining_info_size = remaining_info_size.saturating_sub(1);
                } else {
                    // This might happen if the padding byte is missing right at the end
                    // return Err(Error::Static("Missing padding byte for INFO tag"));
                    eprintln!("Warning: Missing padding byte for INFO tag at end of chunk?");
                    // Tolerate this case
                }
            }
        }
        if remaining_info_size > 0 {
            eprintln!(
                "Warning: {} unparsed bytes left in INFO chunk",
                remaining_info_size
            );
            self.skip_bytes(remaining_info_size as u64)?; // Skip any trailing garbage
        }

        Ok(())
    }

    /// Parses the 'fact' chunk content.
    fn parse_fact_chunk(&mut self, chunk_size: u32) -> Result<Option<ParsedWaveChunk>> {
        if chunk_size < 4 {
            return Err(Error::Static("Invalid 'fact' chunk size (< 4 bytes)"));
        }
        let num_sample_frames = self.read_u32_le()?;

        // Skip any extra data in the fact chunk beyond the first 4 bytes
        let remaining_bytes = chunk_size.saturating_sub(4);
        if remaining_bytes > 0 {
            self.skip_bytes(remaining_bytes as u64)?;
        }

        Ok(Some(ParsedWaveChunk::Fact(FactChunk { num_sample_frames })))
    }
}

/// Options derived from parsing the WAV file header chunks.
#[derive(Debug, Clone)]
struct WavReaderOptions {
    codec_params: CodecParams,
    metadata: HashMap<String, String>, // Simple key-value metadata from INFO
    data_start_pos: u64,               // File offset where audio data begins
    data_end_pos: u64,                 // File offset where audio data ends
    has_fmt_chunk: bool,
    has_data_chunk: bool,
}

/// Reads WAV file headers and provides access to format information, metadata,
/// and the raw audio data stream position.
#[derive(Debug)]
pub struct WavReader<R: Read + Seek + Debug> {
    // We keep the BufReader internally to manage seeking and reading
    buf_reader: BufReader<R>,
    // Store the parsed options
    options: WavReaderOptions,
    // Keep track of the current read position within the data chunk
    data_cursor_pos: u64,
}

impl<R: Read + Seek + Debug> WavReader<R> {
    const RIFF_ID: [u8; 4] = *b"RIFF";
    const WAVE_ID: [u8; 4] = *b"WAVE";
    // Use a reasonably large buffer for efficient disk I/O
    const BUFFER_CAPACITY: usize = 1024 * 64;

    /// Attempts to create a new `WavReader` by parsing the headers of the provided reader `R`.
    ///
    /// # Arguments
    /// * `source`: A reader implementing `Read + Seek + Debug`.
    ///
    /// # Errors
    /// Returns `Error` if the file is not a valid RIFF/WAVE file, if I/O errors occur,
    /// or if the format is unsupported (e.g., missing 'fmt ' or 'data' chunks).
    pub fn try_new(source: R) -> Result<Self> {
        let mut buf_reader = BufReader::with_capacity(Self::BUFFER_CAPACITY, source);

        // 1. Read and verify RIFF header
        let mut riff_id_buf = [0u8; 4];
        buf_reader.read_exact(&mut riff_id_buf)?;
        if riff_id_buf != Self::RIFF_ID {
            return Err(Error::Static("Invalid RIFF identifier"));
        }

        let riff_chunk_size = {
            let mut size_bytes = [0u8; 4];
            buf_reader.read_exact(&mut size_bytes)?;
            u32::from_le_bytes(size_bytes)
        };

        // 2. Read and verify WAVE identifier
        let mut wave_id_buf = [0u8; 4];
        buf_reader.read_exact(&mut wave_id_buf)?;
        if wave_id_buf != Self::WAVE_ID {
            return Err(Error::Static("Invalid WAVE identifier"));
        }

        // Current position is start of chunks (after "RIFF", size, "WAVE")
        let chunks_start_offset = 12u64;

        let mut options = WavReaderOptions {
            codec_params: CodecParams::default(),
            metadata: HashMap::new(),
            data_start_pos: 0,
            data_end_pos: 0,
            has_fmt_chunk: false,
            has_data_chunk: false,
        };

        // 3. Parse internal chunks
        let mut parser = ChunkParser::new(&mut buf_reader, riff_chunk_size, chunks_start_offset);

        // Use a closure to handle parsed chunks and update options
        parser.parse_chunks(|chunk| match chunk {
            ParsedWaveChunk::Format(format_chunk) => {
                Self::handle_format_chunk(&mut options, format_chunk)
            }
            ParsedWaveChunk::Data(data_chunk) => Self::handle_data_chunk(&mut options, data_chunk),
            ParsedWaveChunk::Fact(fact_chunk) => Self::handle_fact_chunk(&mut options, fact_chunk),
            ParsedWaveChunk::List(list_chunk) => Self::handle_list_chunk(&mut options, list_chunk),
        })?;

        // 4. Final Validation
        if !options.has_fmt_chunk {
            return Err(Error::Static("Missing required 'fmt ' chunk"));
        }
        if !options.has_data_chunk {
            // Some tools might create WAVs without a data chunk initially
            eprintln!("Warning: Missing 'data' chunk. No audio data present?");
            // Allow creation, but num_frames will be 0.
        }

        // If num_frames wasn't set by 'fact' or 'data', try calculating from data size if possible
        if options.codec_params.num_frames.is_none() && options.has_data_chunk {
            if let Some(block_align) = options.codec_params.block_align {
                if block_align > 0 {
                    let data_size = options.data_end_pos - options.data_start_pos;
                    options.codec_params.num_frames = Some(data_size / block_align as u64);
                }
            }
        }

        // Initialize the data cursor position
        let initial_data_cursor_pos = options.data_start_pos;

        Ok(WavReader {
            buf_reader,
            options,
            data_cursor_pos: initial_data_cursor_pos,
        })
    }

    /// Processes the parsed `FormatChunk`.
    fn handle_format_chunk(options: &mut WavReaderOptions, chunk: FormatChunk) -> Result<()> {
        if options.has_fmt_chunk {
            eprintln!("Warning: Multiple 'fmt ' chunks found. Using the first one.");
            return Ok(()); // Ignore subsequent fmt chunks
        }

        options.codec_params.sample_rate = Some(chunk.sample_rate);
        options.codec_params.num_channels = chunk.num_channels;
        options.codec_params.bits_per_sample = Some(chunk.bits_per_sample);
        options.codec_params.block_align = Some(chunk.block_align);
        options.codec_params.audio_format = Some(chunk.audio_format);
        options.codec_params.byte_rate = Some(chunk.byte_rate);

        // Basic validation
        if chunk.num_channels == 0 {
            return Err(Error::Unsupported("Number of channels cannot be zero"));
        }
        if chunk.sample_rate == 0 {
            return Err(Error::Unsupported("Sample rate cannot be zero"));
        }
        if chunk.block_align == 0 {
            return Err(Error::Unsupported("Block align cannot be zero"));
        }

        // Determine SampleFormat from bits_per_sample (only for PCM)
        if chunk.audio_format == 1 {
            // 1 = PCM
            options.codec_params.sample_format = match chunk.bits_per_sample {
                8 => Some(SampleFormat::Uint8),
                16 => Some(SampleFormat::Int16),
                24 => Some(SampleFormat::Int24),
                32 => Some(SampleFormat::Int32),
                bps => {
                    return Err(Error::Unsupported(Box::leak(
                        format!("Unsupported bits per sample for PCM: {}", bps).into_boxed_str(),
                    )))
                }
            };
        } else {
            // We store the audio_format but don't support decoding non-PCM yet
            // SampleFormat remains None for non-PCM
            options.codec_params.sample_format = None;
            eprintln!("Warning: Non-PCM audio format ({}) detected. Decoding is not supported, but header info is available.", chunk.audio_format);
        }

        options.has_fmt_chunk = true;
        Ok(())
    }

    /// Processes the parsed `FactChunk`.
    fn handle_fact_chunk(options: &mut WavReaderOptions, chunk: FactChunk) -> Result<()> {
        // The 'fact' chunk often provides the canonical number of sample frames,
        // especially for compressed formats, but can be present for PCM too.
        // Prefer this value if available.
        options.codec_params.num_frames = Some(chunk.num_sample_frames as u64);
        Ok(())
    }

    /// Processes the parsed `DataChunk`.
    fn handle_data_chunk(options: &mut WavReaderOptions, chunk: DataChunk) -> Result<()> {
        if options.has_data_chunk {
            eprintln!("Warning: Multiple 'data' chunks found. Using the first one.");
            return Ok(()); // Ignore subsequent data chunks
        }

        options.data_start_pos = chunk.data_start_pos;
        options.data_end_pos = chunk.data_start_pos + chunk.size as u64;

        // If num_frames hasn't been set by a 'fact' chunk, calculate it from
        // the data chunk size and block alignment (if available and valid).
        if options.codec_params.num_frames.is_none() {
            if let Some(block_align) = options.codec_params.block_align {
                if block_align > 0 {
                    let num_frames = chunk.size as u64 / block_align as u64;
                    // Check for potential truncation due to misaligned data chunk size
                    if (chunk.size as u64 % block_align as u64) != 0 {
                        eprintln!("Warning: Data chunk size ({}) is not a multiple of block align ({}). Some data might be truncated.", chunk.size, block_align);
                    }
                    options.codec_params.num_frames = Some(num_frames);
                } else {
                    eprintln!("Warning: Cannot calculate frame count from data chunk as block align is zero.");
                }
            } else {
                eprintln!("Warning: Cannot calculate frame count from data chunk as 'fmt ' chunk (or block align) is missing or parsed later.");
            }
        }

        options.has_data_chunk = true;
        Ok(())
    }

    /// Processes the parsed `ListChunk`.
    fn handle_list_chunk(options: &mut WavReaderOptions, chunk: ListChunk) -> Result<()> {
        // Add tags to the metadata map. If keys conflict, the last one encountered wins.
        for tag in chunk.tags {
            options.metadata.insert(tag.key, tag.value);
        }
        Ok(())
    }

    // --- Public Accessors ---

    /// Returns the total number of sample frames in the audio data.
    /// A frame contains one sample for each channel.
    pub fn num_frames(&self) -> u64 {
        self.options.codec_params.num_frames.unwrap_or(0)
    }

    /// Returns the sample rate (samples per second per channel).
    pub fn sample_rate(&self) -> u32 {
        self.options.codec_params.sample_rate.unwrap_or(0)
    }

    /// Returns the number of audio channels.
    pub fn num_channels(&self) -> u16 {
        self.options.codec_params.num_channels // Should always be set if fmt is present
    }

    /// Returns the number of bits per sample.
    pub fn bits_per_sample(&self) -> u16 {
        self.options.codec_params.bits_per_sample.unwrap_or(0)
    }

    /// Returns the detected `SampleFormat` (e.g., Int16, Uint8).
    /// Returns `None` if the format is not PCM or couldn't be determined.
    pub fn sample_format(&self) -> Option<SampleFormat> {
        self.options.codec_params.sample_format
    }

    /// Returns the audio format code from the 'fmt ' chunk (e.g., 1 for PCM).
    pub fn audio_format(&self) -> Option<u16> {
        self.options.codec_params.audio_format
    }

    /// Returns the block alignment (bytes per sample frame across all channels).
    pub fn block_align(&self) -> u16 {
        self.options.codec_params.block_align.unwrap_or(0)
    }

    /// Returns the byte rate (average bytes per second).
    pub fn byte_rate(&self) -> u32 {
        self.options.codec_params.byte_rate.unwrap_or(0)
    }

    /// Returns an immutable reference to the metadata key-value map.
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.options.metadata
    }

    /// Returns the total number of individual samples (num_frames * num_channels).
    pub fn num_samples(&self) -> u64 {
        self.num_frames().saturating_mul(self.num_channels() as u64)
    }

    /// Returns the duration of the audio file in seconds.
    pub fn duration_seconds(&self) -> f64 {
        let sr = self.sample_rate();
        if sr > 0 {
            self.num_frames() as f64 / sr as f64
        } else {
            0.0
        }
    }

    // --- Internal Read/Seek Methods needed by Decoder ---

    /// Seeks the internal reader to the beginning of the audio data chunk.
    /// Should only be called once by the decoder.
    fn seek_to_data_start(&mut self) -> Result<()> {
        self.buf_reader
            .seek(SeekFrom::Start(self.options.data_start_pos))?;
        self.data_cursor_pos = self.options.data_start_pos;
        Ok(())
    }

    /// Reads exactly `buffer.len()` bytes from the current position in the data chunk.
    /// Updates the internal `data_cursor_pos`.
    /// Returns `Ok(true)` if successful, `Ok(false)` if EOF reached before filling buffer,
    /// `Err` on I/O error.
    fn read_data_exact(&mut self, buffer: &mut [u8]) -> Result<bool> {
        let bytes_to_read = buffer.len();
        let remaining_data = self
            .options
            .data_end_pos
            .saturating_sub(self.data_cursor_pos);

        if remaining_data == 0 {
            return Ok(false); // EOF already reached
        }

        let actual_read_count = (bytes_to_read as u64).min(remaining_data) as usize;

        if actual_read_count == 0 {
            return Ok(false); // Nothing left to read
        }

        match self.buf_reader.read_exact(&mut buffer[..actual_read_count]) {
            Ok(_) => {
                self.data_cursor_pos += actual_read_count as u64;
                Ok(actual_read_count == bytes_to_read) // True if buffer filled completely
            }
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                // This can happen if remaining_data was > 0 but read_exact failed
                // Update cursor with what might have been read (potentially 0)
                self.data_cursor_pos = self.buf_reader.stream_position()?; // Resync cursor
                Ok(false) // Indicate EOF was hit
            }
            Err(e) => Err(e.into()), // Other I/O errors
        }
    }
    /// Gets the current position within the data chunk.
    fn data_position(&self) -> u64 {
        self.data_cursor_pos
    }

    /// Gets the end position of the data chunk.
    fn data_end_position(&self) -> u64 {
        self.options.data_end_pos
    }
}

// --- Decoder ---

/// Decodes PCM audio data from a `WavReader` into `i32` samples.
pub struct WavDecoder<R: Read + Seek + Debug> {
    reader: WavReader<R>,
    decoder_impl: AudioDecoder, // The actual SIMD/scalar implementation dispatcher
    sample_format: SampleFormat, // Store the format for quick access
    num_channels: usize,        // Store as usize for calculations
    // Alignment required by the chosen decoder (usually 32 for AVX2, 16 for SSE)
    buffer_alignment: RuntimeAlign,
}

impl<R: Read + Seek + Debug> WavDecoder<R> {
    /// Creates a new WAV decoder from a `WavReader`.
    ///
    /// # Errors
    /// Returns `Error::Unsupported` if the audio format is not PCM (format code 1)
    /// or if the sample format (bits per sample) is not supported.
    pub fn try_new(mut reader: WavReader<R>) -> Result<Self> {
        // Ensure the audio format is PCM (1).
        match reader.audio_format() {
            Some(1) => { /* PCM, proceed */ }
            Some(af) => {
                return Err(Error::Unsupported(Box::leak(
                    format!(
                        "Unsupported audio format code: {} (only PCM=1 is supported for decoding)",
                        af
                    )
                    .into_boxed_str(),
                )))
            }
            None => {
                return Err(Error::Unsupported(
                    "Missing audio format code in 'fmt ' chunk",
                ))
            }
        };

        let sample_format = reader.sample_format().ok_or(Error::Unsupported(
            "Unsupported or undetermined sample format for PCM audio",
        ))?;

        // Seek the reader to the start of the actual audio data.
        reader.seek_to_data_start()?;

        let decoder_impl = AudioDecoder::new();
        let buffer_alignment = match decoder_impl {
            AudioDecoder::Avx2(_) => RuntimeAlign::new(32).unwrap(),
            AudioDecoder::Sse42(_) => RuntimeAlign::new(16).unwrap(),
            AudioDecoder::Scalar(_) => RuntimeAlign::new(4).unwrap(), // i32 alignment
        };

        Ok(Self {
            num_channels: reader.num_channels() as usize,
            sample_format,
            reader,
            decoder_impl,
            buffer_alignment,
        })
    }

    /// Returns an iterator that yields packets of decoded audio samples as `&[i32]`.
    /// Each packet contains `PACK_SIZE * num_channels` interleaved samples.
    pub fn packets(&mut self) -> PacketsIterator<R> {
        PacketsIterator::new(
            &mut self.reader,
            &self.decoder_impl,
            self.sample_format,
            self.num_channels,
            self.buffer_alignment,
        )
    }

    // --- Accessors Delegating to WavReader ---

    pub fn num_frames(&self) -> u64 {
        self.reader.num_frames()
    }
    pub fn sample_rate(&self) -> u32 {
        self.reader.sample_rate()
    }
    pub fn num_channels(&self) -> u16 {
        self.reader.num_channels()
    }
    pub fn bits_per_sample(&self) -> u16 {
        self.reader.bits_per_sample()
    }
    pub fn sample_format(&self) -> SampleFormat {
        self.sample_format
    }
    pub fn metadata(&self) -> &HashMap<String, String> {
        self.reader.metadata()
    }
    pub fn num_samples(&self) -> u64 {
        self.reader.num_samples()
    }
    pub fn duration_seconds(&self) -> f64 {
        self.reader.duration_seconds()
    }
}

/// Internal state parameters for the `PacketsIterator`.
struct PacketsIteratorParams {
    bytes_per_sample: usize,
    bytes_per_frame: usize, // bytes_per_sample * num_channels
    // Number of *interleaved* samples to read/write per iteration
    samples_in_packet: usize, // PACK_SIZE * num_channels
    // Number of *bytes* to read/write per iteration
    bytes_in_packet: usize, // samples_in_packet * bytes_per_sample

    // State updated each iteration
    samples_in_current_packet: usize, // Actual number of samples read/decoded in this step
    is_data_available: bool,          // Flag indicating if more data might be available
}

/// An iterator that reads raw audio data in chunks, decodes it into `i32` samples,
/// and yields slices (`&[i32]`) representing packets of audio.
pub struct PacketsIterator<'a, R: Read + Seek + Debug> {
    reader: &'a mut WavReader<R>,
    decoder: &'a AudioDecoder, // Reference to the chosen decoder impl
    sample_format: SampleFormat,
    params: PacketsIteratorParams,
    // Read buffer for raw bytes (aligned)
    rbuffer: AVec<u8>,
    // Write buffer for decoded i32 samples (aligned)
    wbuffer: AVec<i32>,

    _marker: PhantomData<R>, // Keep phantom data if needed, though maybe not here
}

impl<'a, R: Read + Seek + Debug> PacketsIterator<'a, R> {
    fn new(
        reader: &'a mut WavReader<R>,
        decoder: &'a AudioDecoder,
        sample_format: SampleFormat,
        num_channels: usize,
        buffer_alignment: RuntimeAlign,
    ) -> Self {
        let bytes_per_sample = sample_format.bytes_per_sample() as usize;
        let bytes_per_frame = bytes_per_sample * num_channels;
        let samples_in_packet = PACK_SIZE * num_channels;
        let bytes_in_packet = samples_in_packet * bytes_per_sample;

        // Allocate aligned buffers using aligned_vec::avec_rt!
        let rbuffer = AVec::new_aligned(bytes_in_packet, buffer_alignment);
        let wbuffer = AVec::new_aligned(samples_in_packet, buffer_alignment);

        let params = PacketsIteratorParams {
            bytes_per_sample,
            bytes_per_frame,
            samples_in_packet,
            bytes_in_packet,
            samples_in_current_packet: 0,
            is_data_available: true, // Assume data is available initially
        };

        Self {
            reader,
            decoder,
            sample_format,
            params,
            rbuffer,
            wbuffer,
            _marker: PhantomData,
        }
    }
}

impl<'a, R: Read + Seek + Debug> FallibleStreamingIterator for PacketsIterator<'a, R> {
    type Item = [i32]; // Yield slices of the internal buffer
    type Error = Error;

    /// Reads the next chunk of raw data, decodes it, and updates internal state.
    fn advance(&mut self) -> std::result::Result<(), Self::Error> {
        // Check if we already know there's no more data
        if !self.params.is_data_available {
            self.params.samples_in_current_packet = 0;
            return Ok(());
        }

        let current_pos = self.reader.data_position();
        let end_pos = self.reader.data_end_position();

        if current_pos >= end_pos {
            self.params.is_data_available = false;
            self.params.samples_in_current_packet = 0;
            return Ok(());
        }

        let remaining_bytes = end_pos - current_pos;
        let bytes_to_read = remaining_bytes.min(self.params.bytes_in_packet as u64) as usize;

        // Ensure bytes_to_read is a multiple of bytes_per_sample
        // This should generally be true unless the data chunk itself ends mid-sample
        let bytes_to_read_aligned =
            (bytes_to_read / self.params.bytes_per_sample) * self.params.bytes_per_sample;

        if bytes_to_read_aligned == 0 {
            // Not enough bytes left for even one sample
            self.params.is_data_available = false;
            self.params.samples_in_current_packet = 0;
            return Ok(());
        }

        let read_buffer_slice = &mut self.rbuffer[..bytes_to_read_aligned];

        // Read data from the WavReader's internal buffer
        let completed_read = self.reader.read_data_exact(read_buffer_slice)?;

        // Update availability based on whether the read was complete
        self.params.is_data_available =
            completed_read && (bytes_to_read_aligned == self.params.bytes_in_packet);

        // Calculate the number of samples actually read
        self.params.samples_in_current_packet =
            bytes_to_read_aligned / self.params.bytes_per_sample;

        let write_buffer_slice = &mut self.wbuffer[..self.params.samples_in_current_packet];

        // Decode the raw bytes into the i32 write buffer using the chosen decoder
        self.decoder.decode_by_format(
            self.sample_format,
            read_buffer_slice,  // Pass the slice containing read data
            write_buffer_slice, // Pass the slice to write decoded data into
        );

        Ok(())
    }

    /// Returns a reference to the currently decoded packet of samples, if available.
    fn get(&self) -> Option<&Self::Item> {
        if self.params.samples_in_current_packet > 0 {
            // Return a slice of the write buffer containing the valid decoded samples
            Some(&self.wbuffer[..self.params.samples_in_current_packet])
        } else {
            None // No data available in the current packet
        }
    }

    /// Returns the number of items remaining in the iterator. Exact only if num_frames is known.
    fn size_hint(&self) -> (usize, Option<usize>) {
        let frames_total = self.reader.num_frames();
        let bytes_per_frame = self.params.bytes_per_frame as u64;
        if frames_total > 0 && bytes_per_frame > 0 {
            let current_byte_pos = self.reader.data_position();
            let start_byte_pos = self.reader.options.data_start_pos; // Assuming data_start_pos is accurate
            let bytes_processed = current_byte_pos.saturating_sub(start_byte_pos);
            let frames_processed = bytes_processed / bytes_per_frame;
            let frames_remaining = frames_total.saturating_sub(frames_processed);

            // Estimate packets remaining
            let packets_approx = (frames_remaining + PACK_SIZE as u64 - 1) / PACK_SIZE as u64;
            let count = packets_approx as usize;
            (count, Some(count))
        } else {
            (0, None) // Unknown size
        }
    }
}

// --- Decoding Implementations ---

/// Trait defining the interface for audio sample decoding.
trait Decoder {
    // Note: Lifetimes ensure slices are valid, but complexity is low here.
    fn decode_uint8(rbuffer: &[u8], wbuffer: &mut [i32]);
    fn decode_int16(rbuffer: &[u8], wbuffer: &mut [i32]);
    fn decode_int24(rbuffer: &[u8], wbuffer: &mut [i32]);
    fn decode_int32(rbuffer: &[u8], wbuffer: &mut [i32]);
}

/// Dispatches decoding calls to the appropriate implementation (AVX2, SSE4.1, Scalar)
/// based on runtime CPU feature detection.
#[derive(Debug)]
enum AudioDecoder {
    Avx2(Avx2Decoder),
    Sse42(Sse42Decoder), // SSE4.1 often implies SSE2, 3, etc.
    Scalar(ScalarDecoder),
}

impl AudioDecoder {
    /// Creates a new `AudioDecoder`, selecting the best available implementation.
    fn new() -> Self {
        // Check for features in order of preference (most optimized first)
        if is_x86_feature_detected!("avx2") {
            println!("Using AVX2 decoder.");
            AudioDecoder::Avx2(Avx2Decoder)
        } else if is_x86_feature_detected!("sse4.1") {
            println!("Using SSE4.1 decoder.");
            AudioDecoder::Sse42(Sse42Decoder)
        } else {
            println!("Using Scalar decoder.");
            AudioDecoder::Scalar(ScalarDecoder)
        }
    }

    /// Decodes the raw bytes (`rbuffer`) into `i32` samples (`wbuffer`)
    /// according to the specified `SampleFormat`.
    #[inline]
    fn decode_by_format(
        &self,
        format: SampleFormat,
        rbuffer: &[u8],      // Read buffer slice (potentially unaligned)
        wbuffer: &mut [i32], // Write buffer slice (guaranteed aligned by AVec)
    ) {
        let total_samples = wbuffer.len();
        debug_assert_eq!(
            rbuffer.len(),
            total_samples * format.bytes_per_sample() as usize,
            "Buffer length mismatch"
        );

        match self {
            AudioDecoder::Avx2(_) => {
                Self::decode_with_impl::<Avx2Decoder>(format, rbuffer, wbuffer)
            }
            AudioDecoder::Sse42(_) => {
                Self::decode_with_impl::<Sse42Decoder>(format, rbuffer, wbuffer)
            }
            AudioDecoder::Scalar(_) => {
                Self::decode_with_impl::<ScalarDecoder>(format, rbuffer, wbuffer)
            }
        }
    }

    /// Helper function to call the correct method on the chosen Decoder implementation.
    #[inline]
    fn decode_with_impl<D: Decoder>(format: SampleFormat, rbuffer: &[u8], wbuffer: &mut [i32]) {
        match format {
            SampleFormat::Uint8 => D::decode_uint8(rbuffer, wbuffer),
            SampleFormat::Int16 => D::decode_int16(rbuffer, wbuffer),
            SampleFormat::Int24 => D::decode_int24(rbuffer, wbuffer),
            SampleFormat::Int32 => D::decode_int32(rbuffer, wbuffer),
        }
    }
}

// --- Scalar Decoder Implementation ---

#[derive(Debug)]
struct ScalarDecoder;

impl Decoder for ScalarDecoder {
    #[inline]
    fn decode_uint8(rbuffer: &[u8], wbuffer: &mut [i32]) {
        debug_assert_eq!(rbuffer.len(), wbuffer.len());
        for (i, &byte) in rbuffer.iter().enumerate() {
            wbuffer[i] = u8_to_i32(byte);
        }
    }

    #[inline]
    fn decode_int16(rbuffer: &[u8], wbuffer: &mut [i32]) {
        let total_samples = wbuffer.len();
        debug_assert_eq!(rbuffer.len(), total_samples * 2);
        for i in 0..total_samples {
            let start = i * 2;
            wbuffer[i] = i16_to_i32(&rbuffer[start..start + 2]);
        }
    }

    #[inline]
    fn decode_int24(rbuffer: &[u8], wbuffer: &mut [i32]) {
        let total_samples = wbuffer.len();
        debug_assert_eq!(rbuffer.len(), total_samples * 3);
        for i in 0..total_samples {
            let start = i * 3;
            wbuffer[i] = i24_to_i32(&rbuffer[start..start + 3]);
        }
    }

    #[inline]
    fn decode_int32(rbuffer: &[u8], wbuffer: &mut [i32]) {
        let total_samples = wbuffer.len();
        debug_assert_eq!(rbuffer.len(), total_samples * 4);
        for i in 0..total_samples {
            let start = i * 4;
            wbuffer[i] = i32_to_i32(&rbuffer[start..start + 4]);
        }
    }
}

// --- AVX2 Decoder Implementation ---

#[derive(Debug)]
struct Avx2Decoder;

impl Decoder for Avx2Decoder {
    #[target_feature(enable = "avx2")]
    unsafe fn decode_uint8(rbuffer: &[u8], wbuffer: &mut [i32]) {
        decode_uint8_avx2(rbuffer, wbuffer);
    }

    #[target_feature(enable = "avx2")]
    unsafe fn decode_int16(rbuffer: &[u8], wbuffer: &mut [i32]) {
        decode_int16_avx2(rbuffer, wbuffer);
    }

    // No efficient AVX2 for packed 24-bit. Fallback to scalar.
    // #[target_feature(enable = "avx2")] // Keep attribute for consistency?
    fn decode_int24(rbuffer: &[u8], wbuffer: &mut [i32]) {
        // Fallback to scalar implementation
        ScalarDecoder::decode_int24(rbuffer, wbuffer);
    }

    #[target_feature(enable = "avx2")]
    unsafe fn decode_int32(rbuffer: &[u8], wbuffer: &mut [i32]) {
        decode_int32_avx2(rbuffer, wbuffer);
    }
}

// --- SSE4.1 Decoder Implementation ---

#[derive(Debug)]
struct Sse42Decoder; // Assuming SSE4.1 is the target feature level

impl Decoder for Sse42Decoder {
    #[target_feature(enable = "sse4.1")]
    unsafe fn decode_uint8(rbuffer: &[u8], wbuffer: &mut [i32]) {
        decode_uint8_sse41(rbuffer, wbuffer);
    }

    #[target_feature(enable = "sse4.1")]
    unsafe fn decode_int16(rbuffer: &[u8], wbuffer: &mut [i32]) {
        decode_int16_sse41(rbuffer, wbuffer);
    }

    // No efficient SSE for packed 24-bit. Fallback to scalar.
    // #[target_feature(enable = "sse4.1")]
    fn decode_int24(rbuffer: &[u8], wbuffer: &mut [i32]) {
        // Fallback to scalar implementation
        ScalarDecoder::decode_int24(rbuffer, wbuffer);
    }

    #[target_feature(enable = "sse4.1")] // SSE2 is sufficient for i32 load/store
    unsafe fn decode_int32(rbuffer: &[u8], wbuffer: &mut [i32]) {
        decode_int32_sse2(rbuffer, wbuffer); // Use SSE2 version as it's enough
    }
}

// --- Low-Level SIMD Decoding Functions ---

// --- AVX2 Functions ---

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn decode_uint8_avx2(rbuffer: &[u8], wbuffer: &mut [i32]) {
    let total_samples = wbuffer.len();
    let mut i = 0;
    // Process 32 samples (bytes) at a time
    let step = 32;
    let offset = _mm256_set1_epi8(128i8); // Center unsigned byte around zero

    while i + step <= total_samples {
        // Load 32 bytes (unaligned)
        let chunk_ptr = rbuffer.as_ptr().add(i) as *const __m256i;
        let chunk_u8 = _mm256_loadu_si256(chunk_ptr);

        // Subtract 128 to center around 0: u8 -> i8
        let chunk_i8 = _mm256_sub_epi8(chunk_u8, offset);

        // Convert i8 -> i16 (sign extended)
        let chunk_i8_low128 = _mm256_castsi256_si128(chunk_i8); // Lower 16 bytes
        let chunk_i16_low = _mm256_cvtepi8_epi16(chunk_i8_low128); // 16xi16 from lower 16xi8

        let chunk_i8_high128 = _mm256_extracti128_si256::<1>(chunk_i8); // Upper 16 bytes
        let chunk_i16_high = _mm256_cvtepi8_epi16(chunk_i8_high128); // 16xi16 from upper 16xi8

        // Convert i16 -> i32 (sign extended)
        // Low 16xi16 -> Low 8xi32 + High 8xi32
        let chunk_i16_low_low128 = _mm256_castsi256_si128(chunk_i16_low);
        let chunk_i32_0 = _mm256_cvtepi16_epi32(chunk_i16_low_low128); // First 8 i32
        let chunk_i16_low_high128 = _mm256_extracti128_si256::<1>(chunk_i16_low);
        let chunk_i32_1 = _mm256_cvtepi16_epi32(chunk_i16_low_high128); // Next 8 i32

        // High 16xi16 -> Low 8xi32 + High 8xi32
        let chunk_i16_high_low128 = _mm256_castsi256_si128(chunk_i16_high);
        let chunk_i32_2 = _mm256_cvtepi16_epi32(chunk_i16_high_low128); // Next 8 i32
        let chunk_i16_high_high128 = _mm256_extracti128_si256::<1>(chunk_i16_high);
        let chunk_i32_3 = _mm256_cvtepi16_epi32(chunk_i16_high_high128); // Last 8 i32

        // Store 32 results (aligned)
        let out_ptr = wbuffer.as_mut_ptr().add(i) as *mut __m256i;
        _mm256_store_si256(out_ptr.add(0), chunk_i32_0);
        _mm256_store_si256(out_ptr.add(1), chunk_i32_1); // AVX pointers are in units of __m256i
        _mm256_store_si256(out_ptr.add(2), chunk_i32_2);
        _mm256_store_si256(out_ptr.add(3), chunk_i32_3);

        i += step;
    }

    // Process remaining samples scalar
    while i < total_samples {
        wbuffer[i] = u8_to_i32(rbuffer[i]);
        i += 1;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn decode_int16_avx2(rbuffer: &[u8], wbuffer: &mut [i32]) {
    let total_samples = wbuffer.len();
    let mut i = 0;
    // Process 16 samples (32 bytes) at a time
    let step = 16;

    while i + step <= total_samples {
        // Load 16 * i16 = 32 bytes (unaligned)
        let chunk_ptr = rbuffer.as_ptr().add(i * 2) as *const __m256i;
        let chunk_i16 = _mm256_loadu_si256(chunk_ptr);

        // Convert i16 -> i32 (sign extended)
        // Low 8 * i16 -> 8 * i32
        let low_128 = _mm256_castsi256_si128(chunk_i16);
        let chunk_i32_low = _mm256_cvtepi16_epi32(low_128); // First 8 i32

        // High 8 * i16 -> 8 * i32
        let high_128 = _mm256_extracti128_si256::<1>(chunk_i16);
        let chunk_i32_high = _mm256_cvtepi16_epi32(high_128); // Next 8 i32

        // Store 16 results (aligned)
        let out_ptr = wbuffer.as_mut_ptr().add(i) as *mut __m256i;
        _mm256_store_si256(out_ptr.add(0), chunk_i32_low);
        _mm256_store_si256(out_ptr.add(1), chunk_i32_high); // offset by 1 * __m256i = 8 * i32

        i += step;
    }

    // Process remaining samples scalar
    while i < total_samples {
        let start = i * 2;
        wbuffer[i] = i16_to_i32(&rbuffer[start..start + 2]);
        i += 1;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn decode_int32_avx2(rbuffer: &[u8], wbuffer: &mut [i32]) {
    let total_samples = wbuffer.len();
    let mut i = 0;
    // Process 8 samples (32 bytes) at a time
    let step = 8;

    while i + step <= total_samples {
        // Load 8 * i32 = 32 bytes (unaligned)
        let chunk_ptr = rbuffer.as_ptr().add(i * 4) as *const __m256i;
        let chunk_i32 = _mm256_loadu_si256(chunk_ptr);

        // Store 8 results (aligned) - direct copy
        let out_ptr = wbuffer.as_mut_ptr().add(i) as *mut __m256i;
        _mm256_store_si256(out_ptr, chunk_i32);

        i += step;
    }

    // Process remaining samples scalar
    while i < total_samples {
        let start = i * 4;
        wbuffer[i] = i32_to_i32(&rbuffer[start..start + 4]);
        i += 1;
    }
}

// --- SSE4.1 Functions ---

#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn decode_uint8_sse41(rbuffer: &[u8], wbuffer: &mut [i32]) {
    let total_samples = wbuffer.len();
    let mut i = 0;
    // Process 16 samples (bytes) at a time
    let step = 16;
    let offset = _mm_set1_epi8(128i8); // Center unsigned byte around zero

    while i + step <= total_samples {
        // Load 16 bytes (unaligned)
        let chunk_ptr = rbuffer.as_ptr().add(i) as *const __m128i;
        let chunk_u8 = _mm_loadu_si128(chunk_ptr);

        // Subtract 128: u8 -> i8
        let chunk_i8 = _mm_sub_epi8(chunk_u8, offset);

        // Convert i8 -> i16 (sign extended)
        let low_i16 = _mm_cvtepi8_epi16(chunk_i8); // First 8 bytes -> 8xi16
        let high_bytes_shifted = _mm_srli_si128::<8>(chunk_i8); // Shift high 8 bytes down
        let high_i16 = _mm_cvtepi8_epi16(high_bytes_shifted); // Next 8 bytes -> 8xi16

        // Convert i16 -> i32 (sign extended)
        // Low 8xi16 -> Low 4xi32 + High 4xi32
        let chunk_i32_0 = _mm_cvtepi16_epi32(low_i16); // First 4 i32
        let low_i16_high_shifted = _mm_srli_si128::<8>(low_i16); // Shift high 4 i16 down
        let chunk_i32_1 = _mm_cvtepi16_epi32(low_i16_high_shifted); // Next 4 i32

        // High 8xi16 -> Low 4xi32 + High 4xi32
        let chunk_i32_2 = _mm_cvtepi16_epi32(high_i16); // Next 4 i32
        let high_i16_high_shifted = _mm_srli_si128::<8>(high_i16); // Shift high 4 i16 down
        let chunk_i32_3 = _mm_cvtepi16_epi32(high_i16_high_shifted); // Last 4 i32

        // Store 16 results (aligned)
        let out_ptr = wbuffer.as_mut_ptr().add(i) as *mut __m128i;
        _mm_store_si128(out_ptr.add(0), chunk_i32_0);
        _mm_store_si128(out_ptr.add(1), chunk_i32_1);
        _mm_store_si128(out_ptr.add(2), chunk_i32_2);
        _mm_store_si128(out_ptr.add(3), chunk_i32_3);

        i += step;
    }

    // Process remaining samples scalar
    while i < total_samples {
        wbuffer[i] = u8_to_i32(rbuffer[i]);
        i += 1;
    }
}

#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn decode_int16_sse41(rbuffer: &[u8], wbuffer: &mut [i32]) {
    let total_samples = wbuffer.len();
    let mut i = 0;
    // Process 8 samples (16 bytes) at a time
    let step = 8;

    while i + step <= total_samples {
        // Load 8 * i16 = 16 bytes (unaligned)
        let chunk_ptr = rbuffer.as_ptr().add(i * 2) as *const __m128i;
        let chunk_i16 = _mm_loadu_si128(chunk_ptr);

        // Convert i16 -> i32 (sign extended)
        // Low 4 * i16 -> 4 * i32
        let chunk_i32_low = _mm_cvtepi16_epi32(chunk_i16); // First 4 i32

        // High 4 * i16 -> 4 * i32
        let high_half_shifted = _mm_srli_si128::<8>(chunk_i16); // Shift high 4 i16 down
        let chunk_i32_high = _mm_cvtepi16_epi32(high_half_shifted); // Next 4 i32

        // Store 8 results (aligned)
        let out_ptr = wbuffer.as_mut_ptr().add(i) as *mut __m128i;
        _mm_store_si128(out_ptr.add(0), chunk_i32_low);
        _mm_store_si128(out_ptr.add(1), chunk_i32_high);

        i += step;
    }

    // Process remaining samples scalar
    while i < total_samples {
        let start = i * 2;
        wbuffer[i] = i16_to_i32(&rbuffer[start..start + 2]);
        i += 1;
    }
}

// --- SSE2 Function (Sufficient for i32) ---

#[target_feature(enable = "sse2")] // SSE2 is enough for simple load/store
#[inline]
unsafe fn decode_int32_sse2(rbuffer: &[u8], wbuffer: &mut [i32]) {
    let total_samples = wbuffer.len();
    let mut i = 0;
    // Process 4 samples (16 bytes) at a time
    let step = 4;

    while i + step <= total_samples {
        // Load 4 * i32 = 16 bytes (unaligned)
        let chunk_ptr = rbuffer.as_ptr().add(i * 4) as *const __m128i;
        let chunk_i32 = _mm_loadu_si128(chunk_ptr);

        // Store 4 results (aligned) - direct copy
        let out_ptr = wbuffer.as_mut_ptr().add(i) as *mut __m128i;
        _mm_store_si128(out_ptr, chunk_i32);

        i += step;
    }

    // Process remaining samples scalar
    while i < total_samples {
        let start = i * 4;
        wbuffer[i] = i32_to_i32(&rbuffer[start..start + 4]);
        i += 1;
    }
}

// --- Scalar Sample Conversion Helpers ---

/// Converts a single unsigned 8-bit sample to a centered i32 sample.
/// Maps [0, 255] to roughly [-128, 127].
#[inline(always)]
pub fn u8_to_i32(byte: u8) -> i32 {
    // Subtract 128 to center the range around 0
    (byte as i32) - 128
    // For wider range, could multiply/shift: e.g., ((byte as i32) - 128) << 8
}

/// Converts a 2-byte little-endian slice to an i32 sample.
#[inline(always)]
pub fn i16_to_i32(bytes: &[u8]) -> i32 {
    debug_assert_eq!(bytes.len(), 2, "Invalid slice length for i16 conversion");
    // `as i32` performs sign extension automatically
    i16::from_le_bytes(bytes.try_into().unwrap()) as i32
}

/// Converts a 3-byte little-endian slice to an i32 sample with sign extension.
#[inline(always)]
pub fn i24_to_i32(bytes: &[u8]) -> i32 {
    debug_assert_eq!(bytes.len(), 3, "Invalid slice length for i24 conversion");
    // Construct the 24-bit value in the lower bits of a u32
    let val_u24 = (bytes[0] as u32) | ((bytes[1] as u32) << 8) | ((bytes[2] as u32) << 16);

    // Check the sign bit (bit 23) and manually sign extend if necessary
    if (val_u24 & 0x00800000) != 0 {
        // Negative number, extend the sign bit (1s) into the upper byte
        (val_u24 | 0xFF000000) as i32
    } else {
        // Positive number, upper byte is already 0
        val_u24 as i32
    }
    // Alternative using from_le_bytes trick:
    // i32::from_le_bytes([bytes[0], bytes[1], bytes[2], if bytes[2] & 0x80 != 0 { 0xFF } else { 0x00 }])
}

/// Converts a 4-byte little-endian slice to an i32 sample (direct conversion).
#[inline(always)]
pub fn i32_to_i32(bytes: &[u8]) -> i32 {
    debug_assert_eq!(bytes.len(), 4, "Invalid slice length for i32 conversion");
    i32::from_le_bytes(bytes.try_into().unwrap())
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng}; // Use a seeded RNG for reproducible tests
    use std::io::Cursor;

    /// Creates a minimal valid WAV header (RIFF, WAVE, fmt, data) for testing.
    fn minimal_wav_header(
        sample_rate: u32,
        num_channels: u16,
        bits_per_sample: u16,
        data_size: u32, // Size of the data chunk content
    ) -> Vec<u8> {
        let audio_format: u16 = 1; // PCM
        let block_align = num_channels * (bits_per_sample / 8);
        let byte_rate = sample_rate * block_align as u32;
        let fmt_chunk_size: u32 = 16;
        let wave_content_size = 4 /*WAVE*/ + (8 + fmt_chunk_size) + (8 + data_size);
        let riff_chunk_size = wave_content_size; // Total size after "RIFF" and size field

        let mut header = Vec::new();
        // RIFF header (12 bytes)
        header.extend_from_slice(b"RIFF");
        header.extend_from_slice(&riff_chunk_size.to_le_bytes());
        header.extend_from_slice(b"WAVE");

        // fmt chunk (8 bytes header + 16 bytes content)
        header.extend_from_slice(b"fmt ");
        header.extend_from_slice(&fmt_chunk_size.to_le_bytes());
        header.extend_from_slice(&audio_format.to_le_bytes());
        header.extend_from_slice(&num_channels.to_le_bytes());
        header.extend_from_slice(&sample_rate.to_le_bytes());
        header.extend_from_slice(&byte_rate.to_le_bytes());
        header.extend_from_slice(&block_align.to_le_bytes());
        header.extend_from_slice(&bits_per_sample.to_le_bytes());

        // data chunk header (8 bytes)
        header.extend_from_slice(b"data");
        header.extend_from_slice(&data_size.to_le_bytes());

        // The actual data should be appended after this header
        header
    }

    /// Creates random audio data for a given format.
    fn generate_random_data(format: SampleFormat, num_samples: usize, seed: u64) -> Vec<u8> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut data = Vec::with_capacity(num_samples * format.bytes_per_sample() as usize);
        match format {
            SampleFormat::Uint8 => {
                for _ in 0..num_samples {
                    data.push(rng.gen());
                }
            }
            SampleFormat::Int16 => {
                for _ in 0..num_samples {
                    data.extend_from_slice(&rng.gen::<i16>().to_le_bytes());
                }
            }
            SampleFormat::Int24 => {
                for _ in 0..num_samples {
                    let val = rng.gen_range(-(1i32 << 23)..(1i32 << 23)); // Generate within 24-bit signed range
                    let bytes = val.to_le_bytes();
                    data.extend_from_slice(&bytes[0..3]); // Take lower 3 bytes
                }
            }
            SampleFormat::Int32 => {
                for _ in 0..num_samples {
                    data.extend_from_slice(&rng.gen::<i32>().to_le_bytes());
                }
            }
        }
        data
    }

    #[test]
    fn invalid_riff_header() {
        let invalid = b"XXXX\x00\x00\x00\x00WAVEfmt ".to_vec(); // Truncated but wrong ID
        let cursor = Cursor::new(invalid);
        let reader = WavReader::try_new(cursor);
        assert!(reader.is_err());
        match reader {
            Err(Error::Static(msg)) => assert_eq!(msg, "Invalid RIFF identifier"),
            Err(e) => panic!("Expected Static error, got {:?}", e),
            Ok(_) => panic!("Expected error, but got Ok"),
        }
    }

    #[test]
    fn invalid_wave_header() {
        let invalid = b"RIFF\x24\x00\x00\x00XXXXfmt ".to_vec(); // Wrong WAVE ID
        let cursor = Cursor::new(invalid);
        let reader = WavReader::try_new(cursor);
        assert!(reader.is_err());
        match reader {
            Err(Error::Static(msg)) => assert_eq!(msg, "Invalid WAVE identifier"),
            Err(e) => panic!("Expected Static error, got {:?}", e),
            Ok(_) => panic!("Expected error, but got Ok"),
        }
    }

    #[test]
    fn minimal_wav_header_parses_correctly() {
        let data = minimal_wav_header(44100, 2, 16, 0); // 0 data size
        let cursor = Cursor::new(data);
        let reader = WavReader::try_new(cursor).expect("Failed to parse minimal WAV header");
        assert_eq!(reader.sample_rate(), 44100);
        assert_eq!(reader.num_channels(), 2);
        assert_eq!(reader.bits_per_sample(), 16);
        assert_eq!(reader.sample_format(), Some(SampleFormat::Int16));
        assert_eq!(reader.block_align(), 4); // 2 channels * 2 bytes/sample
        assert_eq!(reader.num_frames(), 0); // Since data_size is 0
        assert_eq!(reader.audio_format(), Some(1)); // PCM
    }

    #[test]
    fn truncated_file_returns_eof_error() {
        let truncated = b"RIFF\x08\x00\x00\x00WAV".to_vec(); // Header cut short
        let cursor = Cursor::new(truncated);
        let reader = WavReader::try_new(cursor);
        assert!(reader.is_err());
        match reader {
            // Expect an I/O error, specifically UnexpectedEof
            Err(Error::Io(e)) => assert_eq!(e.kind(), io::ErrorKind::UnexpectedEof),
            Err(e) => panic!("Expected Io(UnexpectedEof), got {:?}", e),
            Ok(_) => panic!("Expected error, but got Ok"),
        }
    }

    #[test]
    fn list_chunk_metadata_parses_correctly() {
        let mut data = minimal_wav_header(48000, 1, 16, 0);
        let list_chunk_size = 4 + (4 + 4 + 5 + 1); // INFO + (IART + size + "Test\0" + pad)
        let info_tag_size = 5; // "Test\0"

        // Add LIST chunk with one tag (IART=Test)
        data.extend_from_slice(b"LIST");
        data.extend_from_slice(&(list_chunk_size as u32).to_le_bytes()); // LIST chunk total size
        data.extend_from_slice(b"INFO"); // LIST type

        // Tag: IART with "Test\0" (odd length, needs padding)
        data.extend_from_slice(b"IART"); // Tag ID
        data.extend_from_slice(&(info_tag_size as u32).to_le_bytes()); // Tag value size
        data.extend_from_slice(b"Test\0"); // Tag value (5 bytes)
        data.push(0); // Padding byte

        // Need to update RIFF size to include the LIST chunk
        let new_riff_size = 4 + (8 + 16) + (8 + list_chunk_size) + (8 + 0); // WAVE + fmt + LIST + data
        data[4..8].copy_from_slice(&new_riff_size.to_le_bytes());

        let cursor = Cursor::new(data);
        let reader = WavReader::try_new(cursor).expect("Failed to parse WAV with INFO chunk");
        let metadata = reader.metadata();

        assert_eq!(metadata.len(), 1);
        assert_eq!(metadata.get("IART"), Some(&"Test".to_string()));
    }

    #[test]
    fn decoder_initializes_for_pcm() {
        let data = minimal_wav_header(32000, 1, 8, 0);
        let cursor = Cursor::new(data);
        let reader = WavReader::try_new(cursor).expect("Failed to parse minimal WAV");
        let decoder = WavDecoder::try_new(reader);
        assert!(decoder.is_ok());
    }

    #[test]
    fn decoder_fails_for_non_pcm() {
        // Create header with audio_format = 3 (IEEE float)
        let mut data = minimal_wav_header(44100, 1, 32, 0); // Use 32-bit for float example
        data[20..22].copy_from_slice(&3u16.to_le_bytes()); // Set audio format to 3 at offset 20

        let cursor = Cursor::new(data);
        let reader = WavReader::try_new(cursor).expect("Should parse non-PCM header");
        let decoder = WavDecoder::try_new(reader);
        assert!(decoder.is_err());
        match decoder {
            Err(Error::Unsupported(msg)) => {
                assert!(msg.contains("Unsupported audio format code: 3"))
            }
            Err(e) => panic!("Expected Unsupported error, got {:?}", e),
            Ok(_) => panic!("Expected error, but got Ok"),
        }
    }

    #[test]
    fn unsupported_bits_per_sample_returns_error() {
        // Create header with bits_per_sample = 20 (unsupported for PCM)
        let mut data = minimal_wav_header(44100, 2, 20, 0); // Use 20 bps
                                                            // Note: block_align and byte_rate in minimal_wav_header will be based on 20/8, might be fractional
                                                            // Let's manually set them correctly if needed, though the check happens on bps
        data[32..34].copy_from_slice(&((2 * (20 / 8)) as u16).to_le_bytes()); // block_align (approx)
        data[34..36].copy_from_slice(&20u16.to_le_bytes()); // bits_per_sample at offset 34

        let cursor = Cursor::new(data);
        let reader = WavReader::try_new(cursor);
        assert!(reader.is_err());
        match reader {
            Err(Error::Unsupported(msg)) => {
                assert!(msg.contains("Unsupported bits per sample for PCM: 20"))
            }
            Err(e) => panic!("Expected Unsupported error, got {:?}", e),
            Ok(_) => panic!("Expected error, but got Ok"),
        }
    }

    #[test]
    fn packets_iterator_no_data_returns_none() {
        let data = minimal_wav_header(16000, 1, 8, 0); // 0 data size
        let cursor = Cursor::new(data);
        let reader = WavReader::try_new(cursor).expect("Failed to parse");
        let mut decoder = WavDecoder::try_new(reader).expect("Failed to create decoder");
        let mut packets = decoder.packets();

        // advance should make `get` return None immediately
        packets
            .advance()
            .expect("Advance should succeed on empty data");
        assert!(
            packets.get().is_none(),
            "get() should return None after advance"
        );

        // Calling next() wraps advance() and get()
        assert!(
            packets.next().unwrap().is_none(),
            "next() should return None"
        );
    }

    #[test]
    fn packets_iterator_partial_packet() {
        let num_channels = 1;
        let sample_format = SampleFormat::Uint8;
        let data_size = PACK_SIZE / 2; // Half a packet worth of samples (bytes since Uint8)
        let mut data = minimal_wav_header(8000, num_channels, 8, data_size as u32);
        let audio_bytes = generate_random_data(sample_format, data_size, 123);
        data.extend_from_slice(&audio_bytes);

        let cursor = Cursor::new(data);
        let reader = WavReader::try_new(cursor).expect("Failed to parse");
        let mut decoder = WavDecoder::try_new(reader).expect("Failed to create decoder");
        let mut packets = decoder.packets();

        // First call to next() should yield the partial packet
        match packets.next().unwrap() {
            Some(p) => {
                assert_eq!(p.len(), data_size, "Packet length should match data size");
                // Verify first sample (optional)
                let expected_first = u8_to_i32(audio_bytes[0]);
                assert_eq!(p[0], expected_first, "First sample mismatch");
            }
            None => panic!("Expected a partial packet, got None"),
        }

        // Second call to next() should yield None (EOF)
        assert!(
            packets.next().unwrap().is_none(),
            "Expected None after consuming partial packet"
        );
    }

    #[test]
    fn packets_iterator_multiple_full_packets() {
        let num_channels = 2;
        let sample_format = SampleFormat::Int16;
        let bits_per_sample = 16;
        let bytes_per_sample = 2;
        let samples_per_packet = PACK_SIZE * num_channels;
        let bytes_per_packet = samples_per_packet * bytes_per_sample;
        let num_packets = 3;
        let total_data_bytes = bytes_per_packet * num_packets;

        let mut data = minimal_wav_header(
            44100,
            num_channels as u16,
            bits_per_sample,
            total_data_bytes as u32,
        );
        let audio_bytes =
            generate_random_data(sample_format, samples_per_packet * num_packets, 456);
        data.extend_from_slice(&audio_bytes);

        let cursor = Cursor::new(data);
        let reader = WavReader::try_new(cursor).expect("Failed to parse");
        let mut decoder = WavDecoder::try_new(reader).expect("Failed to create decoder");
        let mut packets = decoder.packets();
        let mut packet_count = 0;

        while let Some(packet) = packets.next().unwrap() {
            assert_eq!(
                packet.len(),
                samples_per_packet,
                "Packet {} has wrong size",
                packet_count
            );
            // Verify first sample of this packet (optional)
            let packet_start_byte_idx = packet_count * bytes_per_packet;
            let sample_bytes =
                &audio_bytes[packet_start_byte_idx..packet_start_byte_idx + bytes_per_sample];
            let expected_first = i16_to_i32(sample_bytes);
            assert_eq!(
                packet[0], expected_first,
                "First sample mismatch in packet {}",
                packet_count
            );

            packet_count += 1;
        }

        assert_eq!(
            packet_count, num_packets,
            "Incorrect number of packets iterated"
        );
        // Ensure one more call returns None
        assert!(
            packets.next().unwrap().is_none(),
            "Iterator did not end properly"
        );
    }

    // --- Sample Conversion Tests ---
    #[test]
    fn test_sample_conversions() {
        // u8_to_i32: maps 0-255 to -128 to 127
        assert_eq!(u8_to_i32(0), -128);
        assert_eq!(u8_to_i32(128), 0);
        assert_eq!(u8_to_i32(255), 127);

        // i16_to_i32: sign extension
        assert_eq!(i16_to_i32(&i16::MIN.to_le_bytes()), i16::MIN as i32);
        assert_eq!(i16_to_i32(&0i16.to_le_bytes()), 0);
        assert_eq!(i16_to_i32(&i16::MAX.to_le_bytes()), i16::MAX as i32);
        assert_eq!(i16_to_i32(&(-100i16).to_le_bytes()), -100);

        // i24_to_i32: sign extension from 24 bits
        let max_i24 = (1i32 << 23) - 1; // 0x7FFFFF
        let min_i24 = -(1i32 << 23); // -0x800000
        assert_eq!(i24_to_i32(&[0x00, 0x00, 0x00]), 0);
        assert_eq!(i24_to_i32(&[0xFF, 0xFF, 0x7F]), max_i24);
        assert_eq!(i24_to_i32(&[0x00, 0x00, 0x80]), min_i24);
        assert_eq!(i24_to_i32(&[0x01, 0x00, 0x00]), 1);
        assert_eq!(i24_to_i32(&[0xFF, 0xFF, 0xFF]), -1); // [FF, FF, FF] -> -1

        // i32_to_i32: direct mapping
        assert_eq!(i32_to_i32(&i32::MIN.to_le_bytes()), i32::MIN);
        assert_eq!(i32_to_i32(&0i32.to_le_bytes()), 0);
        assert_eq!(i32_to_i32(&i32::MAX.to_le_bytes()), i32::MAX);
    }

    // --- Decoder Implementation Tests ---

    /// Helper to compare scalar and specific SIMD implementation
    fn compare_decoder_impls<D: Decoder>(format: SampleFormat, seed: u64) {
        let num_samples = PACK_SIZE * 2 + 7; // Test full packets + remainder
        let rbuffer_bytes = generate_random_data(format, num_samples, seed);
        let mut wbuffer_scalar = vec![0i32; num_samples];
        let mut wbuffer_simd = vec![0i32; num_samples];

        // Run scalar decoder
        match format {
            SampleFormat::Uint8 => ScalarDecoder::decode_uint8(&rbuffer_bytes, &mut wbuffer_scalar),
            SampleFormat::Int16 => ScalarDecoder::decode_int16(&rbuffer_bytes, &mut wbuffer_scalar),
            SampleFormat::Int24 => ScalarDecoder::decode_int24(&rbuffer_bytes, &mut wbuffer_scalar),
            SampleFormat::Int32 => ScalarDecoder::decode_int32(&rbuffer_bytes, &mut wbuffer_scalar),
        }

        // Run target SIMD decoder (unsafe call within test)
        // Note: Aligned buffers help SIMD, but input might be unaligned.
        // Our functions use loadu/store, so vec should be okay for testing.
        unsafe {
            match format {
                SampleFormat::Uint8 => D::decode_uint8(&rbuffer_bytes, &mut wbuffer_simd),
                SampleFormat::Int16 => D::decode_int16(&rbuffer_bytes, &mut wbuffer_simd),
                // Use scalar for i24 if D doesn't override it
                SampleFormat::Int24 => D::decode_int24(&rbuffer_bytes, &mut wbuffer_simd),
                SampleFormat::Int32 => D::decode_int32(&rbuffer_bytes, &mut wbuffer_simd),
            }
        }

        // Compare results
        assert_eq!(wbuffer_scalar.len(), wbuffer_simd.len());
        for i in 0..num_samples {
            assert_eq!(
                wbuffer_scalar[i], wbuffer_simd[i],
                "Mismatch at index {} for format {:?}",
                i, format
            );
        }
    }

    #[test]
    fn test_avx2_vs_scalar_uint8() {
        if is_x86_feature_detected!("avx2") {
            compare_decoder_impls::<Avx2Decoder>(SampleFormat::Uint8, 101);
        } else {
            println!("Skipping AVX2 Uint8 test: Feature not detected");
        }
    }
    #[test]
    fn test_avx2_vs_scalar_int16() {
        if is_x86_feature_detected!("avx2") {
            compare_decoder_impls::<Avx2Decoder>(SampleFormat::Int16, 102);
        } else {
            println!("Skipping AVX2 Int16 test: Feature not detected");
        }
    }
    #[test]
    fn test_avx2_vs_scalar_int24() {
        // AVX2 decoder uses scalar fallback for Int24, so this should always pass if scalar works
        if is_x86_feature_detected!("avx2") {
            compare_decoder_impls::<Avx2Decoder>(SampleFormat::Int24, 103);
        } else {
            println!("Skipping AVX2 Int24 test: Feature not detected");
        }
    }

    #[test]
    fn test_avx2_vs_scalar_int32() {
        if is_x86_feature_detected!("avx2") {
            compare_decoder_impls::<Avx2Decoder>(SampleFormat::Int32, 104);
        } else {
            println!("Skipping AVX2 Int32 test: Feature not detected");
        }
    }

    #[test]
    fn test_sse41_vs_scalar_uint8() {
        if is_x86_feature_detected!("sse4.1") {
            compare_decoder_impls::<Sse42Decoder>(SampleFormat::Uint8, 201);
        } else {
            println!("Skipping SSE4.1 Uint8 test: Feature not detected");
        }
    }
    #[test]
    fn test_sse41_vs_scalar_int16() {
        if is_x86_feature_detected!("sse4.1") {
            compare_decoder_impls::<Sse42Decoder>(SampleFormat::Int16, 202);
        } else {
            println!("Skipping SSE4.1 Int16 test: Feature not detected");
        }
    }
    #[test]
    fn test_sse41_vs_scalar_int24() {
        // SSE4.1 decoder uses scalar fallback for Int24
        if is_x86_feature_detected!("sse4.1") {
            compare_decoder_impls::<Sse42Decoder>(SampleFormat::Int24, 203);
        } else {
            println!("Skipping SSE4.1 Int24 test: Feature not detected");
        }
    }

    #[test]
    fn test_sse41_vs_scalar_int32() {
        // Uses SSE2 implementation which is sufficient
        if is_x86_feature_detected!("sse2") {
            // Check SSE2 as that's what the function uses
            compare_decoder_impls::<Sse42Decoder>(SampleFormat::Int32, 204);
        } else {
            println!("Skipping SSE4.1(SSE2) Int32 test: Feature not detected");
        }
    }

    // Add more tests from the original list if needed, adapting them to the new structure.
    // E.g., tests for unknown chunks, multiple LIST chunks, fact chunk handling etc.
    #[test]
    fn test_unknown_chunk_skipping() {
        let data_size = 16u32;
        let mut header = minimal_wav_header(44100, 2, 16, data_size);
        let unknown_chunk_size = 4u32;
        let total_extra_size = 8 + unknown_chunk_size; // ID+Size + Content

        // Insert unknown chunk before 'data'
        let data_chunk_start = header.len() - (8 + data_size as usize);
        header.splice(
            data_chunk_start..data_chunk_start,
            [
                b"JUNK".to_vec(),                          // Unknown ID
                unknown_chunk_size.to_le_bytes().to_vec(), // Size
                vec![0u8; unknown_chunk_size as usize],    // Content
            ]
            .concat(),
        );

        // Update RIFF size
        let new_riff_size = u32::from_le_bytes(header[4..8].try_into().unwrap()) + total_extra_size;
        header[4..8].copy_from_slice(&new_riff_size.to_le_bytes());

        let audio_data = generate_random_data(SampleFormat::Int16, (data_size / 2) as usize, 300);
        let mut full_file = header;
        full_file.extend_from_slice(&audio_data);

        let cursor = Cursor::new(full_file);
        let reader = WavReader::try_new(cursor).expect("Should parse even with unknown chunk");

        // Check basic properties are still correct
        assert_eq!(reader.sample_rate(), 44100);
        assert_eq!(reader.num_channels(), 2);
        assert_eq!(
            reader.num_frames(),
            data_size as u64 / reader.block_align() as u64
        ); // Check frames calculated from data chunk
        assert_eq!(
            reader.data_start_pos,
            (12 + 8 + 16 + 8 + unknown_chunk_size + 8) as u64
        ); // Start pos after JUNK

        // Try decoding to ensure data is read correctly
        let mut decoder = WavDecoder::try_new(reader).expect("Decoder failed");
        let mut packets = decoder.packets();
        let first_packet = packets.next().unwrap();
        assert!(first_packet.is_some());
        assert_eq!(first_packet.unwrap().len(), (data_size / 2) as usize); // num_samples = data_size / bytes_per_sample
        assert!(packets.next().unwrap().is_none()); // Should be EOF
    }
}
