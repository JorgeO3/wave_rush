use std::collections::HashMap;
use std::fmt::Debug;
use std::io::{self, BufReader, Read, Seek, SeekFrom};

// ---------- Error Handling ----------

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Static(&'static str),
    #[error("Generic error: {0}")]
    Generic(String),
    #[error(transparent)]
    Io(#[from] io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

// ---------- Core Types ----------

#[derive(Debug, Clone, Copy)]
pub enum SampleFormat {
    Uint8,
    Int16,
    Int24,
    Int32,
}

impl TryFrom<u16> for SampleFormat {
    type Error = Error;

    fn try_from(bits: u16) -> Result<Self> {
        match bits {
            8 => Ok(SampleFormat::Uint8),
            16 => Ok(SampleFormat::Int16),
            24 => Ok(SampleFormat::Int24),
            32 => Ok(SampleFormat::Int32),
            _ => Err(Error::Static("Unsupported sample format")),
        }
    }
}

#[derive(Debug, Default)]
pub struct TimeBase {
    pub numer: u32,
    pub denom: u32,
}

// ---------- Metadata Types ----------

#[derive(Debug, Clone)]
pub struct Tag {
    pub tag_type: TagType,
    pub key: String,
    pub value: String,
}

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

impl From<&[u8; 4]> for TagType {
    fn from(key: &[u8; 4]) -> Self {
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

// ---------- WAV Chunks ----------

#[derive(Debug, Default)]
pub struct FormatChunk {
    audio_format: u16,
    num_channels: u16,
    sample_rate: u32,
    byte_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
}

impl FormatChunk {
    pub fn bytes_per_sample(&self) -> u16 {
        self.bits_per_sample / 8
    }

    fn from_reader<R: Read + Seek + Debug>(
        reader: &mut SourceStream<R>,
        size: usize,
    ) -> Result<Self> {
        if size < 16 {
            return Err(Error::Static("Invalid format chunk size"));
        }

        let chunk = FormatChunk {
            audio_format: reader.read_u16_le()?,
            num_channels: reader.read_u16_le()?,
            sample_rate: reader.read_u32_le()?,
            byte_rate: reader.read_u32_le()?,
            block_align: reader.read_u16_le()?,
            bits_per_sample: reader.read_u16_le()?,
        };

        if size > 16 {
            reader.skip_bytes((size - 16) as i64)?;
        }

        Ok(chunk)
    }
}

#[derive(Debug)]
pub struct ListChunk {
    list_type: [u8; 4],
    tags: Vec<Tag>,
}

impl ListChunk {
    fn from_reader<R: Read + Seek + Debug>(
        reader: &mut SourceStream<R>,
        size: usize,
    ) -> Result<Self> {
        let list_type = reader.read_exact::<4>()?;

        let mut tags = Vec::new();
        if list_type == *b"INFO" {
            Self::parse_info_tags(reader, size - 4, &mut tags)?;
        } else {
            reader.skip_bytes((size - 4) as i64)?;
        }

        Ok(ListChunk { list_type, tags })
    }

    fn parse_info_tags<R: Read + Seek + Debug>(
        reader: &mut SourceStream<R>,
        mut size: usize,
        tags: &mut Vec<Tag>,
    ) -> Result<()> {
        while size >= 8 {
            let key = reader.read_exact::<4>()?;
            let value_size = reader.read_u32_le()? as usize;
            size -= 8;

            let mut value = vec![0u8; value_size];
            reader.read(&mut value)?;
            size -= value_size;

            let tag = Tag {
                tag_type: TagType::from(&key),
                key: String::from_utf8_lossy(&key).into_owned(),
                value: String::from_utf8_lossy(&value)
                    .trim_end_matches('\0')
                    .to_owned(),
            };
            tags.push(tag);

            // Align to even boundary
            if value_size % 2 != 0 && size > 0 {
                reader.skip_bytes(1)?;
                size -= 1;
            }
        }
        Ok(())
    }
}

// ---------- Stream Reader ----------

#[derive(Debug)]
pub struct SourceStream<R: Read + Seek + Debug> {
    reader: BufReader<R>,
    position: u64,
}

impl<R: Read + Seek + Debug> SourceStream<R> {
    const BUFFER_SIZE: usize = 1024 * 16;

    pub fn new(reader: R) -> Self {
        Self {
            reader: BufReader::with_capacity(Self::BUFFER_SIZE, reader),
            position: 0,
        }
    }

    pub fn read_exact<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut buf = [0u8; N];
        self.reader.read_exact(&mut buf)?;
        self.position += N as u64;
        Ok(buf)
    }

    pub fn read_u16_le(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.read_exact()?))
    }

    pub fn read_u32_le(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read_exact()?))
    }

    pub fn skip_bytes(&mut self, offset: i64) -> Result<()> {
        self.reader.seek(SeekFrom::Current(offset))?;
        self.position = self.position.saturating_add(offset as u64);
        Ok(())
    }

    pub fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        let n = self.reader.read(buf)?;
        self.position += n as u64;
        Ok(n)
    }

    pub fn position(&self) -> u64 {
        self.position
    }
}

// ---------- WAV Reader ----------

const MAX_FRAMES_PER_PACKET: u64 = 1024;

#[derive(Debug)]
pub struct WavReader<R: Read + Seek + Debug> {
    stream: SourceStream<R>,
    format: FormatChunk,
    metadata: HashMap<String, String>,
    data_start: u64,
    data_end: u64,
    frames: u64,
}

impl<R: Read + Seek + Debug> WavReader<R> {
    const RIFF_HEADER: [u8; 4] = *b"RIFF";
    const WAVE_HEADER: [u8; 4] = *b"WAVE";

    pub fn open(reader: R) -> Result<Self> {
        let mut stream = SourceStream::new(reader);

        // Validate headers
        if stream.read_exact::<4>()? != Self::RIFF_HEADER {
            return Err(Error::Static("Invalid RIFF header"));
        }

        let file_size = stream.read_u32_le()? as usize;

        if stream.read_exact::<4>()? != Self::WAVE_HEADER {
            return Err(Error::Static("Invalid WAVE header"));
        }

        let mut reader = WavReader {
            stream,
            format: FormatChunk::default(),
            metadata: HashMap::new(),
            data_start: 0,
            data_end: 0,
            frames: 0,
        };

        reader.parse_chunks(file_size)?;
        Ok(reader)
    }

    fn parse_chunks(&mut self, size: usize) -> Result<()> {
        let mut remaining = size;

        while remaining >= 8 {
            let chunk_id = self.stream.read_exact::<4>()?;
            let chunk_size = self.stream.read_u32_le()? as usize;
            remaining = remaining.saturating_sub(8 + chunk_size);

            match &chunk_id {
                b"fmt " => {
                    self.format = FormatChunk::from_reader(&mut self.stream, chunk_size)?;
                }
                b"data" => {
                    self.data_start = self.stream.position();
                    self.data_end = self.data_start + chunk_size as u64;
                    self.frames = chunk_size as u64 / (self.format.block_align as u64);
                    self.stream.skip_bytes(chunk_size as i64)?;
                }
                b"LIST" => {
                    let list = ListChunk::from_reader(&mut self.stream, chunk_size)?;
                    for tag in list.tags {
                        self.metadata.insert(tag.key, tag.value);
                    }
                }
                _ => {
                    self.stream.skip_bytes(chunk_size as i64)?;
                }
            }

            // Align to even boundary
            if chunk_size % 2 != 0 && remaining > 0 {
                self.stream.skip_bytes(1)?;
                remaining -= 1;
            }
        }

        Ok(())
    }

    // Public API

    pub fn sample_format(&self) -> Result<SampleFormat> {
        SampleFormat::try_from(self.format.bits_per_sample)
    }

    pub fn sample_rate(&self) -> u32 {
        self.format.sample_rate
    }

    pub fn channels(&self) -> u16 {
        self.format.num_channels
    }

    pub fn bits_per_sample(&self) -> u16 {
        self.format.bits_per_sample
    }

    pub fn total_frames(&self) -> u64 {
        self.frames
    }

    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    pub fn packets(&mut self) -> impl Iterator<Item = Result<Vec<u8>>> + '_ {
        PacketIterator::new(self)
    }
}

struct PacketIterator<'a, R: Read + Seek + Debug> {
    reader: &'a mut WavReader<R>,
    position: u64,
}

impl<'a, R: Read + Seek + Debug> PacketIterator<'a, R> {
    fn new(reader: &'a mut WavReader<R>) -> Self {
        Self {
            position: reader.data_start,
            reader,
        }
    }
}

impl<'a, R: Read + Seek + Debug> Iterator for PacketIterator<'a, R> {
    type Item = Result<Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.reader.data_end {
            return None;
        }

        let block_size = self.reader.format.block_align as u64;
        let remaining_blocks = (self.reader.data_end - self.position) / block_size;
        let blocks_to_read = remaining_blocks.min(MAX_FRAMES_PER_PACKET);
        let packet_size = blocks_to_read * block_size;

        if packet_size == 0 {
            return None;
        }

        let mut packet = vec![0; packet_size as usize];
        match self.reader.stream.read(&mut packet) {
            Ok(_) => {
                self.position += packet_size;
                Some(Ok(packet))
            }
            Err(e) => Some(Err(e)),
        }
    }
}
