// #![allow(unused)]
use thiserror::Error;
use derive_more::{Add, Sub};
use std::collections::HashMap;
use std::fmt::Debug;
use std::io::{self, BufReader, Read, Seek, SeekFrom};

#[derive(Debug, Error)]
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
#[derive(Debug)]
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

/// Representa la base de tiempo (número racional como numerador y denominador).
#[derive(Debug, Default)]
pub struct TimeBase {
    pub numer: u32,
    pub denom: u32,
}

/// Representa los parámetros del códec para el archivo WAV.
#[derive(Debug, Default)]
struct CodecParams {
    pub sample_rate: Option<u32>,
    pub num_frames: Option<u64>,
    pub sample_format: Option<SampleFormat>,
    pub bits_per_sample: Option<u16>,
    pub num_channels: u16,
    pub max_frames_per_packet: Option<u64>,
    pub frames_per_block: Option<u64>,
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

impl FormatChunk {
    /// Retorna el número de bytes por muestra.
    pub fn bytes_per_sample(&self) -> u16 {
        self.bits_per_sample / 8
    }
}

/// Representa el chunk "LIST" en un archivo WAV.
pub struct ListChunk {
    list_type: [u8; 4],
    length: u32,
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

/// Un flujo de origen con buffer para leer datos de un archivo WAV.
#[derive(Debug)]
pub struct SourceStream<R: Read + Seek + Debug> {
    reader: BufReader<R>,
    abs_pos: u64,
}

impl<R: Read + Seek + Debug> SourceStream<R> {
    const BUFFER_SIZE: usize = 1024 * 16;

    /// Crea un nuevo SourceStream con el lector dado.
    pub fn new(reader: R) -> Self {
        let reader = BufReader::with_capacity(Self::BUFFER_SIZE, reader);
        Self { reader, abs_pos: 0 }
    }

    /// Lee exactamente N bytes del flujo de origen.
    pub fn read_exact<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut result = [0u8; N];
        self.reader.read_exact(&mut result)?;
        self.abs_pos += N as u64;
        Ok(result)
    }

    /// Lee un u16 en formato little-endian.
    pub fn read_u16_le(&mut self) -> Result<u16> {
        let bytes = self.read_exact::<2>()?;
        Ok(u16::from_le_bytes(bytes))
    }

    /// Lee un u32 en formato little-endian.
    pub fn read_u32_le(&mut self) -> Result<u32> {
        let bytes = self.read_exact::<4>()?;
        Ok(u32::from_le_bytes(bytes))
    }

    /// Busca una posición específica en el flujo.
    pub fn seek(&mut self, pos: SeekFrom) -> Result<u64> {
        let new_pos = self.reader.seek(pos)?;
        self.abs_pos = new_pos;
        Ok(new_pos)
    }

    /// Lee datos del flujo de origen en el buffer dado.
    pub fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        let n = self.reader.read(buf)?;
        self.abs_pos += n as u64;
        Ok(n)
    }

    /// Retorna la posición actual en el flujo.
    pub fn position(&self) -> u64 {
        self.abs_pos
    }
}

/// Analiza chunks de un archivo WAV.
struct ChunkParser<'a, R: Read + Seek + Debug> {
    source_stream: &'a mut SourceStream<R>,
    cursor: usize,
    length: usize,
}

impl<'a, R: Read + Seek + Debug> ChunkParser<'a, R> {
    /// Crea un nuevo analizador de chunks con el flujo de origen y longitud dados.
    pub fn new(source_stream: &'a mut SourceStream<R>, length: usize) -> Self {
        Self {
            source_stream,
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
        self.source_stream.seek(SeekFrom::Current(n as i64))?;
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

    /// Itera sobre cada chunk y aplica la función dada a cada uno.
    pub fn parse_chunks<F>(&mut self, mut f: F) -> Result<()>
    where
        F: FnMut(WaveChunk) -> Result<()>,
    {
        while self.cursor + 8 <= self.length {
            // Lee el ID del chunk y su tamaño.
            let chunk_id = self.read_exact::<4>()?;
            let chunk_size = self.read_u32_le()?;
            let chunk_size_usize = chunk_size as usize;

            // Verifica si el tamaño del chunk excede los bytes restantes.
            if self.length - self.cursor < chunk_size_usize {
                return Err(Error::Static("Chunk size exceeds the remaining length"));
            }

            // Procesa el chunk basado en su ID.
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
        let data_position = self.source_stream.position();
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

        Ok(WaveChunk::List(ListChunk {
            list_type,
            length: chunk_size as u32,
            tags,
        }))
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

/// Lector de archivos WAV.
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

    /// Intenta crear un nuevo WavReader analizando las cabeceras del archivo WAV.
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

    pub fn packets(&mut self) -> PacketIterator<R> {
        PacketIterator::new(self)
    }

    pub fn num_packets(&self) -> u64 {
        let num_frames = self.opts.codec_params.num_frames.unwrap_or(0);
        let frames_per_block = self.opts.packet_info.frames_per_block;
        let block_size = self.opts.packet_info.block_size;
        let frames_per_packet = frames_per_block * block_size;
        (num_frames + frames_per_packet - 1) / frames_per_packet
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

    pub fn num_frames_per_block(&self) -> u64 {
        self.opts.packet_info.frames_per_block
    }

    pub fn data_start(&self) -> u64 {
        self.opts.data_start
    }

    pub fn data_end(&self) -> u64 {
        self.opts.data_end
    }
}

pub struct PacketIterator<'a, R: Read + Seek + Debug> {
    reader: &'a mut WavReader<R>,
    current_pos: u64,
}

impl<'a, R: Read + Seek + Debug> PacketIterator<'a, R> {
    fn new(reader: &'a mut WavReader<R>) -> Self {
        let current_pos = reader.opts.data_start;
        Self {
            current_pos,
            reader,
        }
    }
}

impl<'a, R: Read + Seek + Debug> Iterator for PacketIterator<'a, R> {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_pos >= self.reader.opts.data_end {
            return None;
        }

        let opts = &self.reader.opts;
        let block_size = opts.packet_info.block_size;
        let max_blocks_per_packet = opts.packet_info.max_blocks_per_packet;

        let remaining_blocks = (opts.data_end - self.current_pos) / block_size;
        let blocks_to_read = remaining_blocks.min(max_blocks_per_packet);
        let packet_size = blocks_to_read * block_size;

        if packet_size == 0 {
            return None;
        }

        let mut packet = vec![0; packet_size as usize];
        if self.reader.source_stream.read(&mut packet).is_ok() {
            self.current_pos += packet_size;
            Some(packet)
        } else {
            None
        }
    }
}

pub trait Sample:
    Copy
    + Clone
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + Default
    + PartialOrd
    + PartialEq
    + Sized
{
    /// A unique enum value representing the sample format. This constant may be used to dynamically
    /// choose how to process the sample at runtime.
    const FORMAT: SampleFormat;

    /// The effective number of bits of the valid (clamped) sample range. Quantifies the dynamic
    /// range of the sample format in bits.
    const EFF_BITS: u32;

    /// The mid-point value between the maximum and minimum sample value. If a sample is set to this
    /// value it is silent.
    const MID: Self;

    /// If the sample format does not use the full range of the underlying data type, returns the
    /// sample clamped to the valid range. Otherwise, returns the sample unchanged.
    fn clamped(self) -> Self;
}

impl Sample for u8 {
    const FORMAT: SampleFormat = SampleFormat::Uint8;
    const EFF_BITS: u32 = 8;
    const MID: Self = 128;

    #[inline]
    fn clamped(self) -> Self {
        self
    }
}

impl Sample for i16 {
    const FORMAT: SampleFormat = SampleFormat::Int16;
    const EFF_BITS: u32 = 16;
    const MID: Self = 0;

    #[inline]
    fn clamped(self) -> Self {
        self
    }
}

#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Default, Add, Sub)]
struct i24(i32);
impl Sample for i24 {
    const FORMAT: SampleFormat = SampleFormat::Int24;
    const EFF_BITS: u32 = 24;
    const MID: Self = i24(0);

    #[inline]
    fn clamped(self) -> Self {
        i24(self.0.clamp(-8_388_608, 8_388_607))
    }
}

impl Sample for i32 {
    const FORMAT: SampleFormat = SampleFormat::Int32;
    const EFF_BITS: u32 = 32;
    const MID: Self = 0;

    #[inline]
    fn clamped(self) -> Self {
        self
    }
}

struct AudioSpec {
    sample_rate: u32,
    num_channels: u8,
    // bits_per_sample: u16,
    // sample_format: SampleFormat,
    // num_frames: u64,
    // max_frames_per_packet: u64,
    // frames_per_block: u64,
}

struct RawAudioBuffer<S: Sample> {
    buffer: Vec<S>,
    spec: AudioSpec,
    n_frames: usize,
    n_capacity: usize,
}

enum AudioBuffer {
    Uint8(RawAudioBuffer<u8>),
    Int16(RawAudioBuffer<i16>),
    Int24(RawAudioBuffer<i24>),
    Int32(RawAudioBuffer<i32>),
}

impl AudioBuffer {
    fn new(spec: AudioSpec, n_frames: usize, n_capacity: usize) -> Self {
        todo!("Implement AudioBuffer::new");
    }
}

struct WavDecoder {
    params: CodecParams,
    coded_width: u16,
    buffer: AudioBuffer,
}

impl WavDecoder {
    fn try_new(params: &CodecParams) -> Result<Self> {
        let Some(frames) = params.max_frames_per_packet else {
            return Err(Error::Static("max_frames_per_packet not set"));
        };

        let Some(rate) = params.sample_rate else {
            return Err(Error::Static("sample_rate not set"));
        };

        if params.num_channels < 1 || params.num_channels > 2 {
            return Err(Error::Static("Invalid number of channels"));
        }

        let spec = AudioSpec {
            sample_rate: rate,
            num_channels: params.num_channels as u8,
        };

        let coded_width = Some(params.bits_per_sample) else {
            return Err(Error::Static("bits_per_sample not set"));
        };

        todo!()
    }
}
