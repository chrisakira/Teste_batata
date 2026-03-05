use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};
use tracing::{info, warn};

const SAMPLE_RATE: u32 = 16000;

/// Continuous audio capture that feeds samples into a shared ring buffer.
///
/// This replaces the old "capture N seconds" approach. Audio flows
/// continuously from the microphone into a shared buffer that the
/// main loop reads from in frame-sized chunks.
pub struct AudioCapture {
    _stream: cpal::Stream, // kept alive — dropping stops the stream
    buffer: Arc<Mutex<Vec<f32>>>,
    device_sample_rate: u32,
    device_channels: u16,
}

impl AudioCapture {
    /// Start continuous audio capture from the default input device.
    pub fn start() -> anyhow::Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow::anyhow!("No input device found"))?;

        info!("🎤 Using input device: {}", device.name()?);

        // Try to get a config close to 16kHz mono
        let supported = device.supported_input_configs()?;
        let mut best_config = None;

        for range in supported {
            if range.sample_format() == cpal::SampleFormat::F32 {
                // Prefer 16kHz if available, otherwise pick the closest
                if range.min_sample_rate().0 <= SAMPLE_RATE
                    && range.max_sample_rate().0 >= SAMPLE_RATE
                {
                    best_config = Some(range.with_sample_rate(cpal::SampleRate(SAMPLE_RATE)));
                    break;
                } else {
                    best_config = Some(range.with_max_sample_rate());
                }
            }
        }

        let config = best_config
            .map(|c| c.config())
            .unwrap_or(cpal::StreamConfig {
                channels: 1,
                sample_rate: cpal::SampleRate(48000),
                buffer_size: cpal::BufferSize::Default,
            });

        let device_sample_rate = config.sample_rate.0;
        let device_channels = config.channels;

        info!(
            "Audio config: {}Hz, {} channel(s) (will resample to 16kHz mono)",
            device_sample_rate, device_channels
        );

        let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::with_capacity(
            SAMPLE_RATE as usize * 30, // 30 seconds capacity
        )));
        let buffer_clone = buffer.clone();
        let channels = device_channels;
        let source_rate = device_sample_rate;

        let stream = device.build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mut buf = buffer_clone.lock().unwrap();

                // Downmix to mono if needed
                let mono_samples: Vec<f32> = if channels == 1 {
                    data.to_vec()
                } else {
                    data.chunks(channels as usize)
                        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                        .collect()
                };

                // Resample to 16kHz if needed
                if source_rate == SAMPLE_RATE {
                    buf.extend_from_slice(&mono_samples);
                } else {
                    let ratio = SAMPLE_RATE as f64 / source_rate as f64;
                    let output_len = (mono_samples.len() as f64 * ratio).ceil() as usize;
                    for i in 0..output_len {
                        let src_pos = i as f64 / ratio;
                        let src_idx = src_pos.floor() as usize;
                        let frac = src_pos - src_idx as f64;

                        if src_idx + 1 < mono_samples.len() {
                            let sample = mono_samples[src_idx] * (1.0 - frac as f32)
                                + mono_samples[src_idx + 1] * frac as f32;
                            buf.push(sample);
                        } else if src_idx < mono_samples.len() {
                            buf.push(mono_samples[src_idx]);
                        }
                    }
                }
            },
            |err| {
                warn!("Audio stream error: {}", err);
            },
            None,
        )?;

        stream.play()?;

        Ok(Self {
            _stream: stream,
            buffer,
            device_sample_rate,
            device_channels,
        })
    }

    /// Drain all available samples from the buffer.
    ///
    /// This takes ownership of the buffered samples, clearing the internal buffer.
    /// Used by the main loop to feed frames to the VAD.
    pub fn drain_samples(&self) -> Vec<f32> {
        let mut buf = self.buffer.lock().unwrap();
        std::mem::take(&mut *buf)
    }

    /// Read exactly `count` samples, blocking until available.
    /// Returns fewer samples only if draining is faster than production.
    pub fn read_samples(&self, count: usize) -> Vec<f32> {
        let mut buf = self.buffer.lock().unwrap();
        if buf.len() >= count {
            let samples: Vec<f32> = buf.drain(..count).collect();
            samples
        } else {
            let samples = buf.clone();
            buf.clear();
            samples
        }
    }

    /// Get the number of buffered samples.
    pub fn available_samples(&self) -> usize {
        self.buffer.lock().unwrap().len()
    }
}