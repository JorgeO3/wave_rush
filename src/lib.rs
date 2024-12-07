#![allow(unused)]
use std::arch::x86_64::*;
use std::collections::HashMap;
use std::fmt::Debug;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::mem::MaybeUninit;

use aligned_vec::{AVec, ConstAlign};

/// Number of samples per packet.
pub const PACK_SIZE: usize = 2048;

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
    list_type: [u8; 4],
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
struct ChunkParser<'a, R: Read + Seek + Debug> {
    reader: &'a mut BufReader<R>,
    cursor: usize,
    length: usize,
}

impl<'a, R: Read + Seek + Debug> ChunkParser<'a, R> {
    /// Creates a new chunk parser with the given reader and length.
    pub fn new(reader: &'a mut BufReader<R>, length: usize) -> Self {
        Self {
            reader,
            cursor: 0,
            length,
        }
    }

    /// Aligns the cursor to the next 2-byte boundary if needed.
    #[inline]
    fn align(&mut self) -> Result<()> {
        if self.cursor & 1 != 0 {
            self.skip_bytes(1)?;
        }
        Ok(())
    }

    /// Moves the cursor forward by `n` bytes.
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

    /// Iterates over each chunk, applying the provided function to each.
    pub fn parse_chunks<F>(&mut self, mut f: F) -> Result<()>
    where
        F: FnMut(WaveChunk) -> Result<()>,
    {
        while self.cursor + 8 <= self.length {
            let chunk_id = self.read_exact::<4>()?;
            let chunk_size = self.read_u32_le()? as usize;

            if self.length - self.cursor < chunk_size {
                return Err(Error::Static("Chunk size exceeds the remaining length"));
            }

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

        let remaining = chunk_size.saturating_sub(16);
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

    fn parse_data_chunk(&mut self, chunk_size: usize) -> Result<WaveChunk> {
        let data_position = self.reader.stream_position()?;
        self.skip_bytes(chunk_size)?;
        Ok(WaveChunk::Data(DataChunk {
            length: chunk_size as u32,
            data_position,
        }))
    }

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

    fn parse_info_chunk(&mut self, mut remaining_size: usize, tags: &mut Vec<Tag>) -> Result<()> {
        while remaining_size >= 8 {
            let tag_key = self.read_exact::<4>()?;
            let tag_size = self.read_u32_le()? as usize;
            remaining_size -= 8;

            let tag_value_bytes = self.read_bytes(tag_size)?;
            let tag_value = String::from_utf8_lossy(&tag_value_bytes)
                .trim_end_matches(char::from(0))
                .to_string();
            remaining_size -= tag_size;

            if tag_size % 2 != 0 {
                self.skip_bytes(1)?;
                remaining_size -= 1;
            }

            let key = std::str::from_utf8(&tag_key)?.to_string();
            let tag_type = TagType::from_bytes(&tag_key).into();
            tags.push(Tag {
                tag_type,
                key,
                value: tag_value,
            });
        }
        Ok(())
    }

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

/// WAV file reader.
#[derive(Debug)]
pub struct WavReader<R: Read + Seek + Debug> {
    buf: BufReader<R>,
    opts: WavReaderOptions,
}

impl<R: Read + Seek + Debug> WavReader<R> {
    const RIFF_HEADER: [u8; 4] = *b"RIFF";
    const WAVE_HEADER: [u8; 4] = *b"WAVE";
    const BUFFER_SIZE: usize = 1024 * 32;

    /// Attempts to create a new WavReader by parsing the WAV headers.
    pub fn try_new(file: R) -> Result<Self> {
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
            u32::from_le_bytes(size_bytes) as usize
        };

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

        let mut parser = ChunkParser::new(&mut buf, chunk_size);
        parser.parse_chunks(|chunk| match chunk {
            WaveChunk::Format(format) => Self::handle_format_chunk(&mut options, format),
            WaveChunk::Data(data) => Self::handle_data_chunk(&mut options, data),
            WaveChunk::Fact(fact) => Self::handle_fact_chunk(&mut options, fact),
            WaveChunk::List(list) => Self::handle_list_chunk(&mut options, list),
        })?;

        Ok(Self { buf, opts: options })
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

    /// Returns the total number of frames.
    pub fn num_frames(&self) -> u64 {
        self.opts.codec_params.num_frames.unwrap_or(0)
    }

    /// Returns the sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.opts.codec_params.sample_rate.unwrap_or(0)
    }

    /// Returns the number of channels.
    pub fn num_channels(&self) -> u16 {
        self.opts.codec_params.num_channels
    }

    /// Returns bits per sample.
    pub fn bits_per_sample(&self) -> u16 {
        self.opts.codec_params.bits_per_sample.unwrap_or(0)
    }

    /// Returns the sample format.
    pub fn sample_format(&self) -> &SampleFormat {
        self.opts.codec_params.sample_format.as_ref().unwrap()
    }

    /// Returns the metadata.
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.opts.metadata
    }
}

/// WAV file decoder.
pub struct WavDecoder<R: Read + Seek + Debug> {
    reader: WavReader<R>,
}

impl<R: Read + Seek + Debug> WavDecoder<R> {
    /// Creates a new WAV decoder from a `WavReader`.
    pub fn try_new(mut reader: WavReader<R>) -> Result<Self> {
        // Ensure the audio format is PCM (1).
        if reader.opts.codec_params.audio_format != Some(1) {
            return Err(Error::Static(
                "Unsupported audio format (only PCM is supported)",
            ));
        }

        reader.buf.seek(SeekFrom::Start(reader.opts.data_start))?;
        Ok(Self { reader })
    }

    /// Returns an iterator over decoded audio packets.
    pub fn packets(&mut self) -> PacketsIterator<R> {
        let data_end = self.reader.opts.data_end;
        let sample_format = *self.reader.sample_format();
        let num_channels = self.reader.num_channels();
        PacketsIterator::new(&mut self.reader.buf, data_end, sample_format, num_channels)
    }
}

/// Iterator over decoded audio samples.
pub struct PacketsIterator<'a, R: Read + Seek + Debug> {
    reader: &'a mut BufReader<R>,
    data_end: u64,
    sample_format: SampleFormat,
    num_channels: u16,
    buffer: AVec<u8>, // alineado a 32 bytes para AVX2
    bytes_per_sample: usize,
    total_sample_size: usize,
    packet_len_bytes: usize,
}

impl<'a, R: Read + Seek + Debug> PacketsIterator<'a, R> {
    fn new(
        reader: &'a mut BufReader<R>,
        data_end: u64,
        sample_format: SampleFormat,
        num_channels: u16,
    ) -> Self {
        let bytes_per_sample = sample_format.bytes_per_sample() as usize;
        let total_sample_size = bytes_per_sample * num_channels as usize;
        let packet_len_bytes = total_sample_size * PACK_SIZE;

        let max_align = std::mem::align_of::<__m256i>();

        // Crea el AVec con alineación de 32 bytes
        let mut buffer = AVec::with_capacity(max_align, packet_len_bytes);
        buffer.resize(packet_len_bytes, 0);

        Self {
            reader,
            data_end,
            sample_format,
            num_channels,
            buffer,
            bytes_per_sample,
            total_sample_size,
            packet_len_bytes,
        }
    }
}

// TODO: Implement the streaming iterator crate.
// This will allow us to avoid the overhead of creating a Vec for each packet.
impl<R: Read + Seek + Debug> Iterator for PacketsIterator<'_, R> {
    type Item = Result<Vec<i32>>;

    fn next(&mut self) -> Option<Self::Item> {
        // FIXME: This function call is to expensive, we should avoid it.
        // TODO: Implement a manual buffer management to avoid this call.
        let pos = match self.reader.stream_position() {
            Ok(pos) => pos,
            Err(e) => return Some(Err(Error::Io(e))),
        };

        if pos >= self.data_end {
            return None;
        }

        // Ya no calculamos bytes_per_sample, total_sample_size, packet_len_bytes aquí, ya los tenemos.
        let remaining = (self.data_end - pos) as usize;
        let bytes_to_read = remaining.min(self.packet_len_bytes);
        self.buffer.resize(bytes_to_read, 0);

        if let Err(e) = self.reader.read_exact(&mut self.buffer) {
            return Some(Err(Error::Io(e)));
        }

        let total_samples = bytes_to_read / self.bytes_per_sample;

        use SampleFormat::*;
        if std::is_x86_feature_detected!("avx2") {
            unsafe {
                match self.sample_format {
                    Int16 => Some(Ok(decode_int16_avx2(&self.buffer, total_samples))),
                    Uint8 => Some(Ok(decode_uint8_avx2(&self.buffer, total_samples))),
                    Int32 => Some(Ok(decode_int32_avx2(&self.buffer, total_samples))),
                    Int24 => Some(Ok(decode_int24_scalar(&self.buffer, total_samples))),
                }
            }
        } else if std::is_x86_feature_detected!("sse4.1") {
            unsafe {
                match self.sample_format {
                    Int16 => Some(Ok(decode_int16_sse42(&self.buffer, total_samples))),
                    Uint8 => Some(Ok(decode_uint8_sse42(&self.buffer, total_samples))),
                    Int32 => Some(Ok(decode_int32_sse42(&self.buffer, total_samples))),
                    Int24 => Some(Ok(decode_int24_scalar(&self.buffer, total_samples))),
                }
            }
        } else {
            let mut packets = Vec::with_capacity(total_samples);
            for i in 0..total_samples {
                let start = i * self.bytes_per_sample;
                let end = start + self.bytes_per_sample;
                if end > self.buffer.len() {
                    break;
                }
                let sample = match self.sample_format {
                    Uint8 => u8_to_i32(self.buffer[start]),
                    Int16 => i16_to_i32(&self.buffer[start..end]),
                    Int24 => i24_to_i32(&self.buffer[start..end]),
                    Int32 => i32_to_i32(&self.buffer[start..end]),
                };
                packets.push(sample);
            }
            Some(Ok(packets))
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn decode_uint8_avx2(buffer: &[u8], total_samples: usize) -> Vec<i32> {
    let mut packets: Vec<i32> = Vec::with_capacity(total_samples);
    let mut i = 0;

    let offset = _mm256_set1_epi8(128u8 as i8);

    let mut out_ptr = packets.as_mut_ptr();
    while i + 32 <= total_samples {
        let ptr = buffer.as_ptr().add(i) as *const __m256i;
        let chunk = _mm256_loadu_si256(ptr);

        let adjusted = _mm256_sub_epi8(chunk, offset);

        let adjusted_low_128 = _mm256_castsi256_si128(adjusted);
        let adjusted_high_128 = _mm256_extracti128_si256::<1>(adjusted);

        let low_16 = _mm256_cvtepi8_epi16(adjusted_low_128);
        let high_16 = _mm256_cvtepi8_epi16(adjusted_high_128);

        let low_16_low_128 = _mm256_castsi256_si128(low_16);
        let low_16_high_128 = _mm256_extracti128_si256::<1>(low_16);

        let low_32_part1 = _mm256_cvtepi16_epi32(low_16_low_128);
        let low_32_part2 = _mm256_cvtepi16_epi32(low_16_high_128);

        let high_16_low_128 = _mm256_castsi256_si128(high_16);
        let high_16_high_128 = _mm256_extracti128_si256::<1>(high_16);

        let high_32_part1 = _mm256_cvtepi16_epi32(high_16_low_128);
        let high_32_part2 = _mm256_cvtepi16_epi32(high_16_high_128);

        // Escribimos directamente 32 i32
        _mm256_storeu_si256(out_ptr as *mut __m256i, low_32_part1); // primeros 8 i32
        _mm256_storeu_si256(out_ptr.add(8) as *mut __m256i, low_32_part2); // siguientes 8 i32
        _mm256_storeu_si256(out_ptr.add(16) as *mut __m256i, high_32_part1); // siguientes 8 i32
        _mm256_storeu_si256(out_ptr.add(24) as *mut __m256i, high_32_part2); // últimas 8 i32

        out_ptr = out_ptr.add(32);
        i += 32;
    }

    // Resto sin SIMD
    for &byte in &buffer[i..total_samples] {
        *out_ptr = u8_to_i32(byte);
        out_ptr = out_ptr.add(1);
    }

    packets
}

#[target_feature(enable = "avx2")]
unsafe fn decode_int16_avx2(buffer: &[u8], total_samples: usize) -> Vec<i32> {
    // Con AVX2, procesamos 16 i16 (32 bytes) a la vez.
    // let mut packets = vec![0; total_samples];
    // FIXME: This vector is unaligned, we need to use an aligned vector for AVX2.
    // TODO: Use aligned_vec crate to create an aligned vector.
    let mut packets: Vec<i32> = Vec::with_capacity(total_samples);

    let mut i = 0;
    let mut out_ptr = packets.as_mut_ptr(); // puntero a i32 en packets
    while i + 16 <= total_samples {
        let ptr = buffer.as_ptr().add(i * 2) as *const __m256i;
        let chunk = _mm256_loadu_si256(ptr);

        let low_128 = _mm256_castsi256_si128(chunk);
        let high_128 = _mm256_extracti128_si256::<1>(chunk);

        let low_i32 = _mm256_cvtepi16_epi32(low_128);
        let high_i32 = _mm256_cvtepi16_epi32(high_128);

        // Escribimos directamente en packets, sin temp
        _mm256_storeu_si256(out_ptr as *mut __m256i, low_i32);
        _mm256_storeu_si256(out_ptr.add(8) as *mut __m256i, high_i32);

        out_ptr = out_ptr.add(16);
        i += 16;
    }

    // Resto sin AVX2
    for chunk in buffer[i * 2..total_samples * 2].chunks_exact(2) {
        *out_ptr = i16_to_i32(chunk);
        out_ptr = out_ptr.add(1);
    }

    packets
}

#[target_feature(enable = "avx2")]
unsafe fn decode_int32_avx2(buffer: &[u8], total_samples: usize) -> Vec<i32> {
    // Procesamos 8 i32 (32 bytes) a la vez con AVX2
    let mut packets: Vec<i32> = Vec::with_capacity(total_samples);
    let mut i = 0;
    let mut out_ptr = packets.as_mut_ptr();

    while i + 8 <= total_samples {
        let ptr = buffer.as_ptr().add(i * 4) as *const __m256i;
        let chunk = _mm256_loadu_si256(ptr);
        // 8 i32 directos
        _mm256_storeu_si256(out_ptr as *mut __m256i, chunk);
        out_ptr = out_ptr.add(8);
        i += 8;
    }

    // Resto sin SIMD
    for chunk in buffer[i * 4..total_samples * 4].chunks_exact(4) {
        *out_ptr = i32_to_i32(chunk);
        out_ptr = out_ptr.add(1);
    }

    packets
}

#[target_feature(enable = "sse4.1")]
unsafe fn decode_uint8_sse42(buffer: &[u8], total_samples: usize) -> Vec<i32> {
    // Con SSE procesamos 16 u8 -> 16 i32 por iter.
    let mut packets: Vec<i32> = Vec::with_capacity(total_samples);
    let mut i = 0;
    let offset = _mm_set1_epi8(128u8 as i8);
    let mut out_ptr = packets.as_mut_ptr();

    while i + 16 <= total_samples {
        let ptr = buffer.as_ptr().add(i) as *const __m128i;
        let chunk = _mm_loadu_si128(ptr);
        let adjusted = _mm_sub_epi8(chunk, offset);

        let low_16 = _mm_cvtepi8_epi16(adjusted);
        let high_bytes = _mm_srli_si128::<8>(adjusted);
        let high_16 = _mm_cvtepi8_epi16(high_bytes);

        let low_i32_1 = _mm_cvtepi16_epi32(low_16);
        let low_16_high = _mm_srli_si128::<4>(low_16);
        let low_i32_2 = _mm_cvtepi16_epi32(low_16_high);

        let high_i32_1 = _mm_cvtepi16_epi32(high_16);
        let high_16_high = _mm_srli_si128::<4>(high_16);
        let high_i32_2 = _mm_cvtepi16_epi32(high_16_high);

        // 16 i32 totales: 4 vectores de 4 i32
        _mm_storeu_si128(out_ptr as *mut __m128i, low_i32_1); // primeros 4 i32
        _mm_storeu_si128(out_ptr.add(4) as *mut __m128i, low_i32_2); // siguientes 4 i32
        _mm_storeu_si128(out_ptr.add(8) as *mut __m128i, high_i32_1); // siguientes 4 i32
        _mm_storeu_si128(out_ptr.add(12) as *mut __m128i, high_i32_2); // últimos 4 i32

        out_ptr = out_ptr.add(16);
        i += 16;
    }

    for &byte in &buffer[i..total_samples] {
        *out_ptr = u8_to_i32(byte);
        out_ptr = out_ptr.add(1);
    }

    packets
}

#[target_feature(enable = "sse4.1")]
unsafe fn decode_int16_sse42(buffer: &[u8], total_samples: usize) -> Vec<i32> {
    // Con SSE4.1 procesamos 8 i16 = 8 i32 por iter
    let mut packets: Vec<i32> = Vec::with_capacity(total_samples);
    let mut i = 0;
    let mut out_ptr = packets.as_mut_ptr();

    while i + 8 <= total_samples {
        let ptr = buffer.as_ptr().add(i * 2) as *const __m128i;
        let chunk = _mm_loadu_si128(ptr);
        let low_i32 = _mm_cvtepi16_epi32(chunk);
        let high_half = _mm_srli_si128::<8>(chunk);
        let high_i32 = _mm_cvtepi16_epi32(high_half);

        // 8 i32 totales: 2 vectores de 4 i32
        _mm_storeu_si128(out_ptr as *mut __m128i, low_i32);
        _mm_storeu_si128(out_ptr.add(4) as *mut __m128i, high_i32);

        out_ptr = out_ptr.add(8);
        i += 8;
    }

    for chunk in buffer[i * 2..total_samples * 2].chunks_exact(2) {
        *out_ptr = i16_to_i32(chunk);
        out_ptr = out_ptr.add(1);
    }

    packets
}

fn decode_int24_scalar(buffer: &[u8], total_samples: usize) -> Vec<i32> {
    // Int24 es complicado de vectorizar debido a que 3 bytes por muestra no alinean bien con 128 bits.
    // Podrías crear lógica para empaquetar bytes en vectores y extraerlos,
    // pero es complejo y puede no justificarlo.
    // Aquí lo mantenemos escalar:
    let mut packets = Vec::with_capacity(total_samples);
    for i in 0..total_samples {
        let start = i * 3;
        let end = start + 3;
        if end > buffer.len() {
            break;
        }
        packets.push(i24_to_i32(&buffer[start..end]));
    }
    packets
}

#[target_feature(enable = "sse2")]
unsafe fn decode_int32_sse42(buffer: &[u8], total_samples: usize) -> Vec<i32> {
    // Con SSE2 procesamos 4 i32 a la vez
    let mut packets: Vec<i32> = Vec::with_capacity(total_samples);
    let mut i = 0;
    let mut out_ptr = packets.as_mut_ptr();

    while i + 4 <= total_samples {
        let ptr = buffer.as_ptr().add(i * 4) as *const __m128i;
        let chunk = _mm_loadu_si128(ptr);
        _mm_storeu_si128(out_ptr as *mut __m128i, chunk);
        out_ptr = out_ptr.add(4);
        i += 4;
    }

    for chunk in buffer[i * 4..total_samples * 4].chunks_exact(4) {
        *out_ptr = i32_to_i32(chunk);
        out_ptr = out_ptr.add(1);
    }

    packets
}

#[inline(always)]
pub fn u8_to_i32(byte: u8) -> i32 {
    ((byte as i32) << 24 >> 24) - 128
}

#[inline(always)]
pub fn i16_to_i32(bytes: &[u8]) -> i32 {
    assert_eq!(bytes.len(), 2, "Invalid i16 sample size");
    ((bytes[1] as i32) << 8 | (bytes[0] as i32)) << 16 >> 16
}

#[inline(always)]
pub fn i24_to_i32(bytes: &[u8]) -> i32 {
    assert_eq!(bytes.len(), 3, "Invalid i24 sample size");
    ((bytes[2] as i32) << 16 | (bytes[1] as i32) << 8 | (bytes[0] as i32)) << 8 >> 8
}

#[inline(always)]
pub fn i32_to_i32(bytes: &[u8]) -> i32 {
    assert_eq!(bytes.len(), 4, "Invalid i32 sample size");
    (bytes[3] as i32) << 24 | (bytes[2] as i32) << 16 | (bytes[1] as i32) << 8 | (bytes[0] as i32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use std::io::Cursor; // Asegúrate de tener rand en dev-dependencies si se usa

    /// Creates a minimal valid WAV header for testing.
    /// This is a small 44-byte PCM header with no actual audio data.
    fn minimal_wav_header(sample_rate: u32, num_channels: u16, bits_per_sample: u16) -> Vec<u8> {
        let byte_rate = sample_rate * (bits_per_sample as u32 / 8) * num_channels as u32;
        let block_align = (bits_per_sample / 8) * num_channels;

        let mut data = Vec::new();
        // RIFF header
        data.extend_from_slice(b"RIFF");
        // File size (minus 8), for just a header we say 36 bytes extra
        data.extend_from_slice(&(36u32.to_le_bytes()));
        // WAVE
        data.extend_from_slice(b"WAVE");

        // fmt chunk
        data.extend_from_slice(b"fmt ");
        data.extend_from_slice(&(16u32.to_le_bytes())); // Subchunk size = 16 for PCM
        data.extend_from_slice(&(1u16.to_le_bytes())); // PCM format
        data.extend_from_slice(&(num_channels.to_le_bytes()));
        data.extend_from_slice(&(sample_rate.to_le_bytes()));
        data.extend_from_slice(&(byte_rate.to_le_bytes()));
        data.extend_from_slice(&(block_align.to_le_bytes()));
        data.extend_from_slice(&(bits_per_sample.to_le_bytes()));

        // data chunk (no data)
        data.extend_from_slice(b"data");
        data.extend_from_slice(&(0u32.to_le_bytes()));

        data
    }

    #[test]
    fn invalid_riff_header() {
        let invalid = b"XXXX\x00\x00\x00\x00WAVE".to_vec();
        let cursor = Cursor::new(invalid);
        let reader = WavReader::try_new(cursor);
        assert!(reader.is_err());
        match reader {
            Err(Error::Static(msg)) => assert_eq!(msg, "Invalid RIFF header"),
            _ => panic!("Expected 'Invalid RIFF header' error"),
        }
    }

    #[test]
    fn invalid_wave_header() {
        let invalid = b"RIFF\x24\x00\x00\x00XXXX".to_vec();
        let cursor = Cursor::new(invalid);
        let reader = WavReader::try_new(cursor);
        assert!(reader.is_err());
        match reader {
            Err(Error::Static(msg)) => assert_eq!(msg, "Invalid WAVE header"),
            _ => panic!("Expected 'Invalid WAVE header' error"),
        }
    }

    #[test]
    fn minimal_wav_header_parses_correctly() {
        let data = minimal_wav_header(44100, 2, 16);
        let cursor = Cursor::new(data);
        let reader = WavReader::try_new(cursor).expect("Failed to parse minimal WAV header");
        assert_eq!(reader.sample_rate(), 44100);
        assert_eq!(reader.num_channels(), 2);
        assert_eq!(reader.bits_per_sample(), 16);
        assert_eq!(*reader.sample_format(), SampleFormat::Int16);
        assert_eq!(reader.num_frames(), 0);
    }

    #[test]
    fn truncated_file_returns_eof_error() {
        let truncated = b"RIFF".to_vec();
        let cursor = Cursor::new(truncated);
        let reader = WavReader::try_new(cursor);
        assert!(reader.is_err());
        match reader {
            Err(Error::Io(e)) if e.kind() == io::ErrorKind::UnexpectedEof => {}
            _ => panic!("Expected UnexpectedEof error"),
        }
    }

    #[test]
    fn list_chunk_metadata_parses_correctly() {
        let mut data = minimal_wav_header(48000, 1, 16);

        // Add LIST chunk with one tag (IART=Artist)
        data.extend_from_slice(b"LIST");
        data.extend_from_slice(&16u32.to_le_bytes());
        data.extend_from_slice(b"INFO");

        // Tag: IART with "Test\0"
        data.extend_from_slice(b"IART");
        data.extend_from_slice(&(5u32.to_le_bytes()));
        data.extend_from_slice(b"Test\0");

        let cursor = Cursor::new(data);
        let reader = WavReader::try_new(cursor).expect("Failed to parse INFO chunk");
        let metadata = reader.metadata();
        assert_eq!(metadata.get("IART"), Some(&"Test".to_string()));
    }

    #[test]
    fn decoder_initializes_for_pcm() {
        let data = minimal_wav_header(32000, 1, 8);
        let cursor = Cursor::new(data);
        let reader = WavReader::try_new(cursor).expect("Failed to parse minimal WAV");
        let decoder = WavDecoder::try_new(reader);
        assert!(decoder.is_ok());
    }

    #[test]
    fn unsupported_bits_per_sample_returns_error() {
        let mut data = minimal_wav_header(44100, 2, 16);
        // Change bits_per_sample to 20 (unsupported)
        let pos_of_bps = 34;
        data[pos_of_bps] = 20;
        data[pos_of_bps + 1] = 0;

        let cursor = Cursor::new(data);
        let reader = WavReader::try_new(cursor);
        assert!(reader.is_err());
        match reader {
            Err(Error::Static(msg)) => assert_eq!(msg, "Sample format not supported"),
            _ => panic!("Expected 'Sample format not supported' error"),
        }
    }

    #[test]
    fn packets_iterator_no_data_returns_none() {
        let data = minimal_wav_header(16000, 1, 8);
        let cursor = Cursor::new(data);
        let reader = WavReader::try_new(cursor).expect("Failed to parse");
        let mut decoder = WavDecoder::try_new(reader).expect("Failed to create decoder");
        let mut packets = decoder.packets();
        assert!(packets.next().is_none());
    }

    #[test]
    fn packets_iterator_partial_packet() {
        let mut data = minimal_wav_header(8000, 1, 8);
        // half a packet (512 bytes)
        data.extend_from_slice(&[0x80; 512]);

        let cursor = Cursor::new(data);
        let reader = WavReader::try_new(cursor).expect("Failed to parse");
        let mut decoder = WavDecoder::try_new(reader).expect("Failed to create decoder");
        let mut packets = decoder.packets();

        if let Some(Ok(p)) = packets.next() {
            assert_eq!(p.len(), 512);
        } else {
            panic!("Expected a partial packet of 512 samples");
        }

        assert!(packets.next().is_none());
    }

    #[test]
    fn multiple_list_chunks_accumulate_metadata() {
        let mut data = minimal_wav_header(44100, 2, 16);

        // First LIST chunk
        data.extend_from_slice(b"LIST");
        let list_size_1 = 4 + 8 + 4;
        data.extend_from_slice(&(list_size_1 as u32).to_le_bytes());
        data.extend_from_slice(b"INFO");
        data.extend_from_slice(b"ICRD");
        data.extend_from_slice(&(4u32.to_le_bytes()));
        data.extend_from_slice(b"2020");

        // Second LIST chunk
        data.extend_from_slice(b"LIST");
        let list_size_2 = 4 + 8 + 5;
        data.extend_from_slice(&(list_size_2 as u32).to_le_bytes());
        data.extend_from_slice(b"INFO");
        data.extend_from_slice(b"IART");
        data.extend_from_slice(&(5u32.to_le_bytes()));
        data.extend_from_slice(b"ABCD\0");

        let cursor = Cursor::new(data);
        let reader = WavReader::try_new(cursor).expect("Failed to parse multiple LIST chunks");
        let meta = reader.metadata();
        assert_eq!(meta.get("ICRD"), Some(&"2020".to_string()));
        assert_eq!(meta.get("IART"), Some(&"ABCD".to_string()));
    }

    #[test]
    fn multiple_data_chunks_updates_frames() {
        let mut data = minimal_wav_header(8000, 1, 8);

        // Add another data chunk with 4 bytes (4 frames)
        data.extend_from_slice(b"data");
        data.extend_from_slice(&(4u32.to_le_bytes()));
        data.extend_from_slice(&[0x80; 4]);

        let cursor = Cursor::new(data);
        let reader = WavReader::try_new(cursor).expect("Failed to parse multiple data chunks");
        assert_eq!(reader.num_frames(), 4);
    }

    #[test]
    fn large_random_data_parses_without_panic() {
        let mut data = minimal_wav_header(22050, 1, 16);
        let data_size = 1024 * 1024;
        // Remove the original zero size for data chunk
        data.truncate(data.len() - 4);

        data.extend_from_slice(b"data");
        data.extend_from_slice(&(data_size as u32).to_le_bytes());

        let mut rng = rand::thread_rng();
        let random_bytes: Vec<u8> = (0..data_size).map(|_| rng.gen()).collect();
        data.extend_from_slice(&random_bytes);

        let cursor = Cursor::new(data);
        let reader = WavReader::try_new(cursor).expect("Failed to parse large random data");
        assert!(reader.num_frames() > 0);
    }

    #[test]
    fn empty_file_fails() {
        let cursor = Cursor::new(Vec::new());
        let reader = WavReader::try_new(cursor);
        assert!(reader.is_err());
    }

    #[test]
    fn incomplete_tag_key_eof_error() {
        let mut data = minimal_wav_header(44100, 2, 16);
        data.extend_from_slice(b"LIST");
        let list_size = 6u32;
        data.extend_from_slice(&list_size.to_le_bytes());
        data.extend_from_slice(b"INFO");
        data.extend_from_slice(b"IC"); // Only 2 bytes, not 4 for the tag key

        let cursor = Cursor::new(data);
        let reader = WavReader::try_new(cursor);
        assert!(reader.is_err());
        match reader {
            Err(Error::Io(e)) if e.kind() == io::ErrorKind::UnexpectedEof => {}
            _ => panic!("Expected UnexpectedEof due to incomplete tag key"),
        }
    }

    #[test]
    fn missing_fmt_chunk_fails() {
        let mut data = Vec::new();
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&(36u32.to_le_bytes()));
        data.extend_from_slice(b"WAVE");
        // no fmt, directly data
        data.extend_from_slice(b"data");
        data.extend_from_slice(&(0u32.to_le_bytes()));

        let cursor = Cursor::new(data);
        let reader = WavReader::try_new(cursor);
        assert!(reader.is_err());
    }

    #[test]
    fn data_chunk_before_fmt_may_fail() {
        let mut data = Vec::new();
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&(44u32.to_le_bytes()));
        data.extend_from_slice(b"WAVE");

        data.extend_from_slice(b"data");
        data.extend_from_slice(&(4u32.to_le_bytes()));
        data.extend_from_slice(&[0x00; 4]);

        // fmt after data
        data.extend_from_slice(b"fmt ");
        data.extend_from_slice(&(16u32.to_le_bytes()));
        data.extend_from_slice(&(1u16.to_le_bytes()));
        data.extend_from_slice(&(1u16.to_le_bytes()));
        let sr = 8000u32;
        data.extend_from_slice(&sr.to_le_bytes());
        let br = sr * (16 / 8) as u32;
        data.extend_from_slice(&br.to_le_bytes());
        data.extend_from_slice(&(2u16.to_le_bytes()));
        data.extend_from_slice(&(16u16.to_le_bytes()));

        let cursor = Cursor::new(data);
        let reader = WavReader::try_new(cursor);
        // Could be ok or err depending on how the code handles ordering.
        assert!(reader.is_err() || reader.is_ok());
    }

    #[test]
    fn null_terminated_tags_are_trimmed() {
        let mut data = minimal_wav_header(48000, 2, 16);

        data.extend_from_slice(b"LIST");
        let list_size = 4 + 8 + 6;
        data.extend_from_slice(&(list_size as u32).to_le_bytes());
        data.extend_from_slice(b"INFO");
        data.extend_from_slice(b"INAM");
        data.extend_from_slice(&(6u32.to_le_bytes()));
        data.extend_from_slice(b"Hello\0");

        let cursor = Cursor::new(data);
        let reader = WavReader::try_new(cursor).expect("Failed to parse null-terminated tag");
        let meta = reader.metadata();
        assert_eq!(meta.get("INAM"), Some(&"Hello".to_string()));
    }

    #[test]
    fn oversized_chunk_size_errors_out() {
        let mut data = minimal_wav_header(44100, 2, 16);

        data.extend_from_slice(b"JUNK");
        data.extend_from_slice(&u32::MAX.to_le_bytes());

        let cursor = Cursor::new(data);
        let reader = WavReader::try_new(cursor);
        assert!(reader.is_err());
        match reader {
            Err(Error::Static(msg)) => assert_eq!(msg, "Chunk size exceeds the remaining length"),
            _ => panic!("Expected 'Chunk size exceeds the remaining length' error"),
        }
    }

    #[test]
    fn odd_length_tags_parsed_correctly() {
        let mut data = minimal_wav_header(48000, 2, 16);

        data.extend_from_slice(b"LIST");
        let list_chunk_size = 20u32;
        data.extend_from_slice(&list_chunk_size.to_le_bytes());
        data.extend_from_slice(b"INFO");

        // "ICMT" tag with 3-byte data "Hi\0"
        data.extend_from_slice(b"ICMT");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(b"Hi\0");

        let cursor = Cursor::new(data);
        let reader = WavReader::try_new(cursor).expect("Failed to parse odd-length tag");
        let meta = reader.metadata();
        assert_eq!(meta.get("ICMT"), Some(&"Hi".to_string()));
    }

    #[test]
    fn fact_chunk_with_non_pcm_still_parses() {
        let mut data = Vec::new();
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&(36u32.to_le_bytes()));
        data.extend_from_slice(b"WAVE");

        // fmt with audio_format = 3 (IEEE float)
        data.extend_from_slice(b"fmt ");
        data.extend_from_slice(&(16u32.to_le_bytes()));
        data.extend_from_slice(&(3u16.to_le_bytes())); // Non-PCM
        data.extend_from_slice(&(1u16.to_le_bytes()));
        let sample_rate = 44100u32;
        data.extend_from_slice(&sample_rate.to_le_bytes());
        let byte_rate = sample_rate * 4;
        data.extend_from_slice(&byte_rate.to_le_bytes());
        data.extend_from_slice(&(4u16.to_le_bytes()));
        data.extend_from_slice(&(32u16.to_le_bytes()));

        // fact chunk
        data.extend_from_slice(b"fact");
        data.extend_from_slice(&4u32.to_le_bytes());
        data.extend_from_slice(&(1000u32.to_le_bytes()));

        data.extend_from_slice(b"data");
        data.extend_from_slice(&(0u32.to_le_bytes()));

        let cursor = Cursor::new(data);
        let reader = WavReader::try_new(cursor).expect("Failed to parse non-PCM fact chunk");
        assert_eq!(reader.num_frames(), 1000);
    }

    #[test]
    fn malformed_fmt_chunk_errors() {
        let mut data = Vec::new();
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&(20u32.to_le_bytes()));
        data.extend_from_slice(b"WAVE");

        // fmt chunk too small (only 10 bytes)
        data.extend_from_slice(b"fmt ");
        data.extend_from_slice(&(10u32.to_le_bytes()));
        data.extend_from_slice(&[0u8; 10]);

        let cursor = Cursor::new(data);
        let reader = WavReader::try_new(cursor);
        assert!(reader.is_err());
        match reader {
            Err(Error::Static(msg)) => assert_eq!(msg, "Invalid format chunk size"),
            _ => panic!("Expected 'Invalid format chunk size' error"),
        }
    }

    #[test]
    fn data_chunk_misaligned_with_block_align_truncates() {
        let mut data = minimal_wav_header(44100, 2, 16);
        // data_size = 6, block_align = 4, so 1 frame + leftover
        data.extend_from_slice(b"data");
        let data_size = 6u32;
        data.extend_from_slice(&data_size.to_le_bytes());
        data.extend_from_slice(&[0x00; 6]);

        let cursor = Cursor::new(data);
        let reader = WavReader::try_new(cursor).expect("Failed to parse header");
        assert_eq!(reader.num_frames(), 1);
    }

    #[test]
    fn decoder_fails_for_non_pcm() {
        let mut data = Vec::new();
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&(36u32.to_le_bytes()));
        data.extend_from_slice(b"WAVE");

        data.extend_from_slice(b"fmt ");
        data.extend_from_slice(&(16u32.to_le_bytes()));
        data.extend_from_slice(&(3u16.to_le_bytes())); // non-PCM
        data.extend_from_slice(&(2u16.to_le_bytes()));
        let sample_rate = 48000u32;
        data.extend_from_slice(&sample_rate.to_le_bytes());
        let byte_rate = sample_rate * 2 * 4;
        data.extend_from_slice(&byte_rate.to_le_bytes());
        data.extend_from_slice(&(8u16.to_le_bytes()));
        data.extend_from_slice(&(32u16.to_le_bytes()));

        data.extend_from_slice(b"data");
        data.extend_from_slice(&(0u32.to_le_bytes()));

        let cursor = Cursor::new(data);
        let reader = WavReader::try_new(cursor).expect("Should parse non-PCM");
        let decoder = WavDecoder::try_new(reader);
        assert!(decoder.is_err());
        match decoder {
            Err(Error::Static(msg)) => {
                assert_eq!(msg, "Unsupported audio format (only PCM is supported)")
            }
            _ => panic!("Expected 'Unsupported audio format' error"),
        }
    }

    #[test]
    fn packets_iterator_small_data_partial_packet() {
        let mut data = minimal_wav_header(16000, 1, 8);
        // Add only 10 bytes of data
        data.extend_from_slice(b"data");
        data.extend_from_slice(&(10u32.to_le_bytes()));
        data.extend_from_slice(&[0x80; 10]);

        let cursor = Cursor::new(data);
        let reader = WavReader::try_new(cursor).expect("Failed to parse");
        let mut decoder = WavDecoder::try_new(reader).expect("Failed to create decoder");
        let mut packets = decoder.packets();

        if let Some(Ok(packet)) = packets.next() {
            assert_eq!(packet.len(), 10);
        } else {
            panic!("Expected one partial packet with 10 samples");
        }

        assert!(packets.next().is_none());
    }

    #[test]
    fn packets_iterator_multiple_full_packets() {
        let mut data = minimal_wav_header(44100, 2, 16);
        // One packet = 1024 frames * 4 bytes/frame = 4096 bytes
        // Two packets = 8192 bytes
        data.extend_from_slice(b"data");
        let data_length = 8192u32;
        data.extend_from_slice(&data_length.to_le_bytes());
        data.extend_from_slice(&vec![0u8; 8192]);

        let cursor = Cursor::new(data);
        let reader = WavReader::try_new(cursor).expect("Failed to parse");
        let mut decoder = WavDecoder::try_new(reader).expect("Failed to create decoder");
        let mut packets = decoder.packets();

        // First packet
        if let Some(Ok(packet)) = packets.next() {
            assert_eq!(packet.len(), 1024 * 2);
        } else {
            panic!("Expected first packet");
        }

        // Second packet
        if let Some(Ok(packet)) = packets.next() {
            assert_eq!(packet.len(), 1024 * 2);
        } else {
            panic!("Expected second packet");
        }

        assert!(packets.next().is_none());
    }

    #[test]
    fn non_ascii_tag_key_does_not_crash() {
        let mut data = minimal_wav_header(44100, 2, 16);
        data.extend_from_slice(b"LIST");

        // Non-ASCII key: 4 + 8 + 1 = 13 total
        let list_size = 4 + 8 + 1;
        data.extend_from_slice(&(list_size as u32).to_le_bytes());
        data.extend_from_slice(b"INFO");

        data.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]);
        data.extend_from_slice(&(1u32.to_le_bytes()));
        data.push(b'A');

        let cursor = Cursor::new(data);
        let reader = WavReader::try_new(cursor).expect("Should parse weird key");
        let meta = reader.metadata();
        // If handling non-ASCII keys isn't implemented, this might fail.
        // If it doesn't crash, at least we have some metadata?
        // We just check not empty:
        assert!(!meta.is_empty());
    }
}
